"""Glean usage, split per person.

Chats are the consumption measure. Active days are the activity measure.
Slug drops @gendigital.com. Seniority and org come from title.
Division is L1 Division. Output goes to glean_output, separate from Copilot.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
XLSX_PATH = ROOT / "GleanUsage.xlsx"
OUT_DIR = ROOT / "glean_output"

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
CLUSTER_NAMES = ["Light", "Steady", "Heavy"]
CLUSTER_COLORS = ["#8aa0b4", "#0f6cbd", "#c43e1c"]


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


def load_usage(path: Path = XLSX_PATH) -> pd.DataFrame:
    df = pd.read_excel(path)
    df["chats"] = pd.to_numeric(df["Chats in period"], errors="coerce").fillna(0)
    df["active_days"] = pd.to_numeric(df["Days active on Assistant in period"], errors="coerce").fillna(0)
    df["summaries"] = pd.to_numeric(df["Summarizations in period"], errors="coerce").fillna(0)
    df["ai_answers"] = pd.to_numeric(df["AI Answers in period"], errors="coerce").fillna(0)
    df["slug"] = df["Email"].map(slug_from_email)
    df["division"] = df["L1 Division"].fillna("Unknown").astype(str).str.strip()
    df["title"] = df["Division Title"].fillna(df["Source Title"])
    df["org"] = df["title"].map(infer_org)
    df["seniority"] = pd.Categorical(df["title"].map(infer_seniority), SENIORITY_ORDER, ordered=True)
    df["level_group"] = pd.Categorical(df["seniority"].astype(str).map(GROUP_OF), GROUP_ORDER, ordered=True)
    return df.dropna(subset=["slug"]).reset_index(drop=True)


def fit_clusters(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    X = StandardScaler().fit_transform(np.log1p(frame[["chats", "active_days"]].to_numpy(dtype=float)))
    scores = {}
    for k in range(2, 6):
        labels = KMeans(n_clusters=k, n_init=20, random_state=42).fit_predict(X)
        scores[k] = silhouette_score(X, labels)
    best_k = max(scores, key=scores.get)
    labels = KMeans(n_clusters=best_k, n_init=20, random_state=42).fit_predict(X)
    frame = frame.copy()
    frame["cluster_id"] = labels
    ranked = frame.groupby("cluster_id")["chats"].median().sort_values().index.tolist()
    names = (CLUSTER_NAMES + ["Segment 4", "Segment 5"])[:best_k]
    frame["usage_cluster"] = frame["cluster_id"].map({old: names[i] for i, old in enumerate(ranked)})
    return frame, best_k


def bootstrap_median(values: np.ndarray, draws: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    samples = [np.median(rng.choice(values, size=len(values), replace=True)) for _ in range(draws)]
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(lo), float(hi)


def per_head_table(frame: pd.DataFrame, column: str, order: list[str]) -> pd.DataFrame:
    company = frame["chats"].sum() / len(frame)
    rows = []
    for name in order:
        part = frame[frame[column].astype(str) == name]
        if part.empty:
            continue
        values = part["chats"].to_numpy(dtype=float)
        lo, hi = bootstrap_median(values)
        rows.append(
            {
                column: name,
                "people": int(len(part)),
                "headcount_share_pct": round(100 * len(part) / len(frame), 1),
                "chat_share_pct": round(100 * values.sum() / frame["chats"].sum(), 1),
                "chats_per_person": round(values.mean(), 0),
                "vs_company": round(values.mean() / company, 2),
                "typical_person": round(float(np.median(values)), 0),
                "typical_range_low": round(lo, 0),
                "typical_range_high": round(hi, 0),
            }
        )
    return pd.DataFrame(rows)


def save_graphs(frame: pd.DataFrame, groups: pd.DataFrame, bands: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    company = frame["chats"].mean()

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    colors = ["#0f6cbd", "#5b8def", "#c43e1c"]
    bars = ax.bar(groups["level_group"], groups["chats_per_person"], color=colors, width=0.62)
    ax.axhline(company, color="#111827", linestyle="--", linewidth=1.1, label=f"Company per person ({company:.0f})")
    for bar, row in zip(bars, groups.itertuples(index=False)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            row.chats_per_person + 4,
            f"{row.chats_per_person:.0f}\n{row.vs_company:.0%} of company\nn={row.people}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylim(0, groups["chats_per_person"].max() * 1.35)
    ax.set_ylabel("Glean chats per person")
    ax.set_title("Glean usage per person, not by group size")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_per_person_by_group.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.4))
    y = np.arange(len(bands))
    ax.hlines(y, bands["typical_range_low"], bands["typical_range_high"], color="#8aa0b4", linewidth=6)
    ax.scatter(bands["typical_person"], y, s=70, color="#0f6cbd", zorder=3, label="Typical person")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{name}  (n={n})" for name, n in zip(bands["seniority"], bands["people"])])
    ax.set_xlabel("Chats for a typical person (line = 95% range)")
    ax.set_title("Glean: typical person by title band")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_typical_person_range.png", dpi=170)
    plt.close(fig)

    div = (
        frame.groupby("division")
        .agg(people=("slug", "nunique"), chats_per_person=("chats", "mean"))
        .sort_values("chats_per_person", ascending=True)
    )
    colors = ["#c43e1c" if name == "Global Customer Organization" else "#0f6cbd" for name in div.index]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(div.index, div["chats_per_person"], color=colors)
    for i, (people, value) in enumerate(zip(div["people"], div["chats_per_person"])):
        ax.text(value + 2, i, f"n={people}", va="center", fontsize=8, color="#333333")
    ax.axvline(company, color="#111827", linestyle="--", linewidth=1.1, label="Company average")
    ax.set_xlabel("Chats per person")
    ax.set_title("Global Customer Organization uses Glean the most per person")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_department_per_person.png", dpi=170)
    plt.close(fig)


def write_results(frame: pd.DataFrame, best_k: int, groups: pd.DataFrame, bands: pd.DataFrame) -> None:
    biggest_group = groups.sort_values("people", ascending=False).iloc[0]
    by_div = frame.groupby("division")["chats"].mean().sort_values(ascending=False)
    top_div = by_div.index[0]
    top_n = int((frame["division"] == top_div).sum())
    top_per = float(by_div.iloc[0])
    lead = groups[groups["level_group"] == "Leadership"].iloc[0]
    ic = groups[groups["level_group"] == "IC"].iloc[0]
    lines = [
        "# Glean usage, per person",
        "",
        "Source: `GleanUsage.xlsx`. Slug drops `@gendigital.com`. Consumption is chats in the period. Division is L1 Division.",
        "",
        (
            f"The largest group is {biggest_group['level_group']} at {biggest_group['people']:.0f} of {len(frame)} people. "
            f"The department that uses Glean the most is {top_div}: {top_n} people, {top_per:.0f} chats per person, "
            f"against a company average of {frame['chats'].mean():.0f}. "
            f"Finance & Business Performance has more people, but each person there uses less. "
            f"Leadership is {lead['people']:.0f} people and uses {lead['chats_per_person']:.0f} chats each, "
            f"about the same as an IC at {ic['chats_per_person']:.0f}."
        ),
        "",
        "## Three headcount groups",
        "",
        groups.to_markdown(index=False),
        "",
        "## Title bands",
        "",
        bands.to_markdown(index=False),
        "",
        "Graphs: `01_per_person_by_group.png`, `02_typical_person_range.png`, `03_department_per_person.png`.",
    ]
    (OUT_DIR / "results.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if OUT_DIR.exists():
        for path in OUT_DIR.iterdir():
            if path.is_file():
                path.unlink()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_usage()
    frame, best_k = fit_clusters(frame)
    groups = per_head_table(frame, "level_group", GROUP_ORDER)
    bands = per_head_table(frame, "seniority", SENIORITY_ORDER)
    save_graphs(frame, groups, bands)
    write_results(frame, best_k, groups, bands)
    frame[
        ["slug", "division", "org", "level_group", "seniority", "title", "chats", "active_days", "usage_cluster"]
    ].to_csv(OUT_DIR / "people.csv", index=False)
    groups.to_csv(OUT_DIR / "per_person_groups.csv", index=False)
    bands.to_csv(OUT_DIR / "per_person_bands.csv", index=False)
    print(f"people={len(frame)} k={best_k} chats_per_person={frame['chats'].mean():.1f}")
    print(groups.to_string(index=False))
    print("wrote", OUT_DIR)


if __name__ == "__main__":
    main()
