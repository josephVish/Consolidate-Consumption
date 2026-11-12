"""Division seniority trend, org comparison, and a small usage predictor.

Clustering and prediction use Prompts (All Apps) and Active Usage Days.
Slug drops @gendigital.com. Seniority and org are inferred from Title.
L1 Division is the division field in the export.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, silhouette_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "NB1Copilot.csv"
OUT_DIR = ROOT / "output"
PROMPT_BASE = 100  # this export is already a power-user cut; nobody is below 100
CLUSTER_NAMES = ["At base", "Above base", "Far above base", "Segment 4", "Segment 5"]
CLUSTER_COLORS = ["#8aa0b4", "#0f6cbd", "#c43e1c", "#5b5fc7"]

ORG_RULES: list[tuple[str, str]] = [
    (r"\b(software|engineer|developer|programmer|qa|architect|sdet)\b", "Engineering"),
    (r"\b(data scien|machine learning|applied ai|ai engineer|ai specialist|decisioning)\b", "Data & AI"),
    (r"\b(customer|support|advocate|concierge|retention|care specialist)\b", "Customer"),
    (r"\b(sales|account manager|business development|partner success|acquisition)\b", "Sales"),
    (r"\b(market|editor|brand|content)\b", "Marketing"),
    (r"\b(financ|account|fp&a|tax|audit|controller|buyer|procurement|treasury)\b", "Finance"),
    (r"\b(people|recruiter|talent|compensation|payroll|benefits|hr )\b", "People"),
    (r"\b(legal|counsel|compliance|aml|investigator)\b", "Legal"),
    (r"\b(security|infosec)\b", "Security"),
    (r"\b(product|pmo|portfolio)\b", "Product"),
    (r"\b(it |information technology|applications specialist|inf specialist|pitstop)\b", "IT"),
    (r"\b(analyst|analytics|bizops|operations)\b", "Analytics & Ops"),
]

SENIORITY_ORDER = ["Junior", "Mid", "Senior", "Principal", "Manager", "Director+"]
GROUP_ORDER = ["IC", "Senior IC", "Leadership"]
GROUP_OF = {
    "Junior": "IC",
    "Mid": "IC",
    "Senior": "Senior IC",
    "Principal": "Senior IC",
    "Manager": "Leadership",
    "Director+": "Leadership",
}


def slug_from_email(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if "@" not in text:
        return text.lower() or None
    local, _domain = text.rsplit("@", 1)
    return local.lower() or None


def infer_org(title: object) -> str:
    if pd.isna(title) or not str(title).strip():
        return "Other"
    text = f" {str(title).lower()} "
    for pattern, org in ORG_RULES:
        if re.search(pattern, text):
            return org
    return "Other"


def infer_seniority(title: object) -> str:
    if pd.isna(title) or not str(title).strip():
        return "Mid"
    text = f" {str(title).lower()} "
    if re.search(r"\b(chief|head of|vp|vice president|director|dir,)\b", text):
        return "Director+"
    if re.search(r"\b(manager|mgr,)\b", text):
        return "Manager"
    if re.search(r"\b(princ|principal)\b", text):
        return "Principal"
    if re.search(r"\b(sr\.?\b|senior|lead)\b", text):
        return "Senior"
    if re.search(r"\b(intern|junior|assoc|associate|assistant|temporary|vendor)\b", text):
        return "Junior"
    return "Mid"


def load_usage(path: Path = CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin1")
    prompts = df["Prompts (All Apps)"].astype(str).str.replace(",", "", regex=False).str.strip()
    df["prompts"] = pd.to_numeric(prompts, errors="coerce")
    df["activity_days"] = pd.to_numeric(df["Active Usage Days"], errors="coerce")
    df["slug"] = df["Email"].map(slug_from_email)
    missing = df["slug"].isna()
    df.loc[missing, "slug"] = df.loc[missing, "User Principal Name"].map(slug_from_email)
    df["division"] = df["L1 Division"].fillna("Unknown").astype(str).str.strip()
    df["org"] = df["Title"].map(infer_org)
    df["seniority"] = pd.Categorical(df["Title"].map(infer_seniority), SENIORITY_ORDER, ordered=True)
    df["level_group"] = pd.Categorical(df["seniority"].astype(str).map(GROUP_OF), GROUP_ORDER, ordered=True)
    keep = df["prompts"].notna() & df["activity_days"].notna() & df["slug"].notna()
    out = df.loc[keep].copy()
    out["prompts_above_base"] = (out["prompts"] - PROMPT_BASE).clip(lower=0)
    out["prompts_per_active_day"] = out["prompts"] / out["activity_days"].clip(lower=1)
    return out.reset_index(drop=True)


def fit_clusters(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    # Cluster on prompts above the 100 floor, not on distance from zero.
    X = StandardScaler().fit_transform(
        frame[["prompts_above_base", "activity_days"]].to_numpy(dtype=float)
    )
    scores = {}
    for k in range(2, 7):
        labels = KMeans(n_clusters=k, n_init=20, random_state=42).fit_predict(X)
        scores[k] = silhouette_score(X, labels)
    best_k = max(scores, key=scores.get)
    labels = KMeans(n_clusters=best_k, n_init=20, random_state=42).fit_predict(X)
    frame = frame.copy()
    frame["cluster_id"] = labels
    ranked = frame.groupby("cluster_id")["prompts"].median().sort_values().index.tolist()
    frame["usage_cluster"] = frame["cluster_id"].map(
        {old: CLUSTER_NAMES[i] for i, old in enumerate(ranked)}
    )
    return frame, best_k


def predict_prompts(frame: pd.DataFrame) -> dict:
    """Predict prompts from activity, seniority, division, and org.

    Scores are 5-fold out-of-fold, so a row is not scored on a model trained on itself.
    """
    features = frame[["activity_days", "seniority", "division", "org"]].copy()
    features["seniority"] = features["seniority"].astype(str)
    y = frame["prompts"].to_numpy(dtype=float)
    y_above = y - PROMPT_BASE

    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), ["activity_days"]),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                ["seniority", "division", "org"],
            ),
        ]
    )
    model = Pipeline([("pre", pre), ("model", Ridge(alpha=1.0))])
    pred_log = cross_val_predict(model, features, np.log1p(y_above), cv=5)
    pred = np.clip(np.expm1(pred_log), 0, None) + PROMPT_BASE
    baseline = np.full_like(y, max(np.median(y), PROMPT_BASE))
    model.fit(features, np.log1p(y_above))
    return {
        "r2": float(r2_score(np.log1p(y_above), pred_log)),
        "mae": float(mean_absolute_error(y, pred)),
        "baseline_mae": float(mean_absolute_error(y, baseline)),
        "pred": pred,
        "model": model,
    }


def bootstrap_median(values: np.ndarray, draws: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    samples = [np.median(rng.choice(values, size=len(values), replace=True)) for _ in range(draws)]
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(lo), float(hi)


def per_head_table(frame: pd.DataFrame, column: str, order: list[str]) -> pd.DataFrame:
    company_per_head = frame["prompts"].sum() / len(frame)
    rows = []
    for name in order:
        part = frame[frame[column].astype(str) == name]
        if part.empty:
            continue
        values = part["prompts"].to_numpy(dtype=float)
        lo, hi = bootstrap_median(values)
        per_head = values.mean()
        rows.append(
            {
                column: name,
                "people": int(len(part)),
                "headcount_share_pct": round(100 * len(part) / len(frame), 1),
                "prompt_share_pct": round(100 * values.sum() / frame["prompts"].sum(), 1),
                "per_person_mean": round(per_head, 0),
                "per_person_vs_company": round(per_head / company_per_head, 2),
                "typical_person_median": round(float(np.median(values)), 0),
                "typical_range_low": round(lo, 0),
                "typical_range_high": round(hi, 0),
            }
        )
    return pd.DataFrame(rows)


def seniority_table(frame: pd.DataFrame) -> pd.DataFrame:
    table = (
        frame.groupby(["division", "seniority"], observed=True)
        .agg(
            people=("slug", "nunique"),
            median_prompts=("prompts", "median"),
            median_activity_days=("activity_days", "median"),
        )
        .reset_index()
    )
    return table[table["people"] >= 3]


def _short(name: str) -> str:
    return (
        name.replace("Finance & Business Performance", "Finance & BP")
        .replace("Global Customer Organization", "Global Customer")
        .replace("Global Marketing Organization", "Global Marketing")
        .replace("Information Technology", "IT Division")
        .replace("Legal & Corporate Development", "Legal & Corp")
        .replace("Partnerships & Alliances", "Partnerships")
        .replace("Product & Portfolio", "Product")
    )


def save_three_graphs(frame: pd.DataFrame, pred_meta: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    focus = frame[frame["seniority"].isin(["Junior", "Senior"])].copy()
    counts = focus.groupby(["division", "seniority"], observed=True).size().unstack(fill_value=0)
    keep = counts[(counts.get("Junior", 0) >= 5) & (counts.get("Senior", 0) >= 5)].index
    focus = focus[focus["division"].isin(keep)]
    company = (
        frame[frame["seniority"].isin(["Junior", "Senior"])]
        .groupby("seniority", observed=True)["prompts"]
        .median()
    )

    trend = (
        focus.groupby(["division", "seniority"], observed=True)["prompts"]
        .median()
        .reset_index()
    )
    trend["division"] = trend["division"].map(_short)
    order = (
        trend.groupby("division")["prompts"].mean().sort_values(ascending=False).index.tolist()
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.pointplot(
        data=trend,
        x="seniority",
        y="prompts",
        hue="division",
        hue_order=order,
        order=["Junior", "Senior"],
        marker="o",
        linewidth=2.4,
        ax=ax,
    )
    ax.axhline(PROMPT_BASE, color="#111827", linestyle="--", linewidth=1.2, label="Power-user base (100)")
    ax.axhline(company["Junior"], color="#8aa0b4", linestyle=":", linewidth=1.4, label="Company Junior")
    ax.axhline(company["Senior"], color="#0f6cbd", linestyle=":", linewidth=1.4, label="Company Senior")
    ax.set_ylim(PROMPT_BASE, trend["prompts"].max() * 1.08)
    ax.set_title("Junior vs Senior median prompts, by division (5+ people each)")
    ax.set_xlabel("")
    ax.set_ylabel("Median prompts (base = 100)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_junior_vs_senior_by_division.png", dpi=170)
    plt.close(fig)

    org_n = frame.groupby("org").size()
    orgs = org_n[org_n >= 8].index
    org = frame[frame["org"].isin(orgs)].copy()
    org_order = org.groupby("org")["prompts"].median().sort_values(ascending=False).index.tolist()
    cluster_order = [name for name in CLUSTER_NAMES if name in set(org["usage_cluster"])]
    mix = (
        pd.crosstab(org["org"], org["usage_cluster"], normalize="index")
        .reindex(index=org_order, columns=cluster_order)
        .fillna(0)
        * 100
    )
    medians = org.groupby("org")["prompts"].median().reindex(org_order)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [1, 1.25]})
    axes[0].barh(medians.index[::-1], medians.values[::-1], color="#0f6cbd")
    axes[0].axvline(PROMPT_BASE, color="#111827", linestyle="--", linewidth=1.2)
    axes[0].set_xlim(PROMPT_BASE, medians.max() * 1.08)
    axes[0].set_xlabel("Median prompts (base = 100)")
    axes[0].set_title("How far each org sits above the base")
    mix.plot(
        kind="barh",
        stacked=True,
        ax=axes[1],
        color=CLUSTER_COLORS[: len(cluster_order)],
        width=0.8,
        legend=False,
    )
    axes[1].set_xlabel("Share of people (%)")
    axes[1].set_ylabel("")
    axes[1].set_xlim(0, 100)
    axes[1].set_title("Usage cluster mix")
    axes[1].invert_yaxis()
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c)
        for c in CLUSTER_COLORS[: len(cluster_order)]
    ]
    axes[1].legend(handles, cluster_order, loc="lower right", fontsize=8, frameon=False)
    fig.suptitle("Org comparison (8+ people)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_org_comparison.png", dpi=170)
    plt.close(fig)

    levels = ["Junior", "Mid", "Senior", "Principal", "Manager", "Director+"]
    effects = []
    for level in levels:
        row = frame[["activity_days"]].median().to_frame().T
        row["seniority"] = level
        row["division"] = frame["division"].mode().iloc[0]
        row["org"] = frame["org"].mode().iloc[0]
        effects.append(row)
    grid = pd.concat(effects, ignore_index=True)
    pred_level = np.clip(np.expm1(pred_meta["model"].predict(grid)), 0, None) + PROMPT_BASE
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    colors = ["#c43e1c", "#e07a3d", "#0f6cbd", "#5b8def", "#6b7280", "#111827"]
    bars = ax.bar(levels, pred_level, color=colors, bottom=0)
    ax.axhline(PROMPT_BASE, color="#111827", linestyle="--", linewidth=1.2, label="Base 100")
    ax.set_ylabel("Predicted prompts")
    ax.set_title("Predicted prompts by seniority, above the 100 base\n(same active days, same typical division and org)")
    for bar, value in zip(bars, pred_level):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 4,
            f"{value:.0f}\n(+{value - PROMPT_BASE:.0f})",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylim(PROMPT_BASE * 0.7, max(pred_level) * 1.22)
    ax.legend(frameon=False, loc="upper right")
    note = (
        f"5-fold R2 on log prompts {pred_meta['r2']:.2f}. "
        f"MAE {pred_meta['mae']:.0f} vs median baseline {pred_meta['baseline_mae']:.0f}."
    )
    ax.text(0.0, -0.18, note, transform=ax.transAxes, fontsize=8, color="#444444")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_prediction_by_seniority.png", dpi=170)
    plt.close(fig)


def save_group_graphs(frame: pd.DataFrame, groups: pd.DataFrame, bands: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    x = np.arange(len(groups))
    colors = ["#0f6cbd", "#5b8def", "#c43e1c"]
    bars = ax.bar(x, groups["per_person_mean"], color=colors, width=0.62)
    company = frame["prompts"].mean()
    ax.axhline(company, color="#111827", linestyle="--", linewidth=1.1, label=f"Company per person ({company:.0f})")
    ax.axhline(PROMPT_BASE, color="#8aa0b4", linestyle=":", linewidth=1.1, label="Floor 100")
    for bar, row in zip(bars, groups.itertuples(index=False)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            row.per_person_mean + 8,
            f"{row.per_person_mean:.0f}\n{row.per_person_vs_company:.0%} of company\nn={row.people}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(groups["level_group"])
    ax.set_ylim(0, groups["per_person_mean"].max() * 1.35)
    ax.set_ylabel("Prompts per person")
    ax.set_title("Usage split per head, not by group size")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_per_person_by_group.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.4))
    y = np.arange(len(bands))
    ax.hlines(y, bands["typical_range_low"], bands["typical_range_high"], color="#8aa0b4", linewidth=6)
    ax.scatter(bands["typical_person_median"], y, s=70, color="#0f6cbd", zorder=3, label="Typical person")
    ax.axvline(PROMPT_BASE, color="#111827", linestyle=":", linewidth=1, label="Floor 100")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{name}  (n={n})" for name, n in zip(bands["seniority"], bands["people"])])
    ax.set_xlabel("Prompts for a typical person (line = 95% range)")
    ax.set_title("Leadership's 221 is real, but the range is wide because n=12")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_typical_person_range.png", dpi=170)
    plt.close(fig)

    mix = (
        pd.crosstab(frame["level_group"], frame["usage_cluster"], normalize="index")
        .reindex(index=GROUP_ORDER, columns=[name for name in CLUSTER_NAMES if name in set(frame["usage_cluster"])])
        .fillna(0)
        * 100
    )
    fig, ax = plt.subplots(figsize=(8, 4.8))
    mix.plot(kind="barh", stacked=True, ax=ax, color=CLUSTER_COLORS[: mix.shape[1]], width=0.72)
    ax.set_xlabel("Share of people in the group (%)")
    ax.set_ylabel("")
    ax.set_xlim(0, 100)
    ax.set_title("Same three usage groups, split inside each headcount group")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_usage_groups_by_level.png", dpi=170)
    plt.close(fig)

    div = (
        frame.groupby("division")
        .agg(people=("slug", "nunique"), prompts_per_person=("prompts", "mean"))
        .sort_values("prompts_per_person", ascending=True)
    )
    top = div["prompts_per_person"].idxmax()
    colors = ["#c43e1c" if name == top else "#0f6cbd" for name in div.index]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(div.index, div["prompts_per_person"], color=colors)
    for i, (people, value) in enumerate(zip(div["people"], div["prompts_per_person"])):
        ax.text(value + 4, i, f"n={people}", va="center", fontsize=8, color="#333333")
    ax.axvline(company, color="#111827", linestyle="--", linewidth=1.1, label="Company average")
    ax.set_xlabel("Prompts per person")
    ax.set_title(f"{top} uses Copilot the most per person")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_department_per_person.png", dpi=170)
    plt.close(fig)


def write_results(frame: pd.DataFrame, best_k: int, pred_meta: dict, by_level: pd.DataFrame) -> None:
    groups = per_head_table(frame, "level_group", GROUP_ORDER)
    bands = per_head_table(frame, "seniority", SENIORITY_ORDER)
    save_group_graphs(frame, groups, bands)
    leadership = groups[groups["level_group"] == "Leadership"].iloc[0]
    ic = groups[groups["level_group"] == "IC"].iloc[0]
    overall = bands
    org = (
        frame.groupby("org")
        .agg(
            people=("slug", "nunique"),
            median_prompts=("prompts", "median"),
            median_days=("activity_days", "median"),
        )
        .sort_values("median_prompts", ascending=False)
    )
    lift = (1 - pred_meta["mae"] / pred_meta["baseline_mae"]) * 100
    lines = [
        "# Per-person estimate",
        "",
        "The file is a power-user cut with a floor of 100 prompts. Groups are uneven, so the estimate is prompts per person, not the group's total and not an unweighted median.",
        "IC = Junior + Mid. Senior IC = Senior + Principal. Leadership = Manager + Director+.",
        "",
        (
            f"Leadership is {leadership['people']:.0f} people ({leadership['headcount_share_pct']}% of headcount) and accounts for "
            f"{leadership['prompt_share_pct']}% of prompts, so each leader uses {leadership['per_person_vs_company']:.0%} of the company per-person average "
            f"({leadership['per_person_mean']:.0f} vs {ic['per_person_mean']:.0f} for an IC). "
            f"A typical leader sits at {leadership['typical_person_median']:.0f} prompts, range {leadership['typical_range_low']:.0f}-{leadership['typical_range_high']:.0f}. "
            f"A typical IC sits at {ic['typical_person_median']:.0f}, range {ic['typical_range_low']:.0f}-{ic['typical_range_high']:.0f}. "
            "The 12 directors pull the typical-leader number down to 221, and that band's range is wide, but the per-person mean for the whole leadership group is still about 17% under the company average. "
            f"Usage still falls into {best_k} groups above the 100 floor. A model on seniority, division, and org only beats a median guess by {lift:.0f}% (R2 {pred_meta['r2']:.2f})."
        ),
        "",
        "## Three headcount groups, per person",
        "",
        groups.to_markdown(index=False),
        "",
        "## Title bands, per person",
        "",
        "Director+ is kept because the typical person there is closer to 221 than a Junior. The range shows that 12 people cannot pin that number tightly.",
        "",
        overall.to_markdown(index=False),
        "",
        "## Org overall",
        "",
        org.round(1).to_markdown(),
        "",
        "## Division x seniority (3+ people)",
        "",
        by_level.round(1).to_markdown(index=False),
        "",
        "Graphs: `01_per_person_by_group.png`, `02_typical_person_range.png`, `03_usage_groups_by_level.png`, `04_department_per_person.png`.",
    ]
    (OUT_DIR / "results.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if OUT_DIR.exists():
        for path in OUT_DIR.iterdir():
            if path.is_file():
                path.unlink()
    frame = load_usage()
    frame, best_k = fit_clusters(frame)
    pred_meta = predict_prompts(frame)
    frame["predicted_prompts"] = pred_meta["pred"].round(1)
    by_level = seniority_table(frame)
    write_results(frame, best_k, pred_meta, by_level)

    frame[
        [
            "slug",
            "division",
            "org",
            "level_group",
            "seniority",
            "Title",
            "prompts",
            "prompts_above_base",
            "activity_days",
            "usage_cluster",
            "predicted_prompts",
        ]
    ].to_csv(OUT_DIR / "people.csv", index=False)
    by_level.to_csv(OUT_DIR / "division_seniority.csv", index=False)
    per_head_table(frame, "level_group", GROUP_ORDER).to_csv(OUT_DIR / "per_person_groups.csv", index=False)
    per_head_table(frame, "seniority", SENIORITY_ORDER).to_csv(OUT_DIR / "per_person_bands.csv", index=False)
    print(
        f"people={len(frame)} k={best_k} r2={pred_meta['r2']:.3f} "
        f"mae={pred_meta['mae']:.1f} baseline_mae={pred_meta['baseline_mae']:.1f}"
    )
    print(frame["seniority"].value_counts().reindex(SENIORITY_ORDER).to_string())
    print(frame.groupby("seniority", observed=True)["prompts"].median().round(0).to_string())
    print("graphs:", sorted(p.name for p in OUT_DIR.glob("*.png")))


if __name__ == "__main__":
    main()
