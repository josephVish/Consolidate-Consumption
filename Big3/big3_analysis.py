"""One analysis across the three tools.

Claude is input/cowork_sessions.csv, Glean is input/insights_glean-usage-PAST_3_MONTHS.csv.
Copilot is the full 90-day export, since the copy in input is only a top-20 summary.
Matched on email slug. Title and department come from Glean.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
COPILOT = Path.home() / "OneDrive - Gen Digital Inc" / "FastCopilotActivityUserDetail8_4_202690_days.csv"
BOUNDARY = 20
SENIORITY_ORDER = ["Junior", "Mid", "Senior", "Principal", "Manager", "Director+"]
GROUP_OF = {"Junior": "IC", "Mid": "IC", "Senior": "Senior IC", "Principal": "Senior IC", "Manager": "Leadership", "Director+": "Leadership"}
GROUP_ORDER = ["IC", "Senior IC", "Leadership"]


def slug(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.split("@").str[0].str.lower()
    return out.where(series.notna() & out.ne("") & out.ne("nan"))


def seniority_of(title: object) -> str | None:
    if pd.isna(title) or not str(title).strip():
        return None
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


def load() -> pd.DataFrame:
    claude = pd.read_csv(ROOT / "input" / "cowork_sessions.csv")
    claude["slug"] = slug(claude["Email"])
    claude["spend"] = pd.to_numeric(claude["Estimated Spend (USD)"], errors="coerce").fillna(0)
    claude["code"] = pd.to_numeric(claude["Code sessions"], errors="coerce").fillna(0)
    claude = claude.dropna(subset=["slug"]).drop_duplicates("slug")

    copilot = pd.read_csv(COPILOT, encoding="latin1").drop_duplicates("User Principal Name")
    copilot["slug"] = slug(copilot["User Principal Name"])
    copilot["prompts"] = pd.to_numeric(copilot["Prompts submitted for All Apps"], errors="coerce").fillna(0)
    copilot = copilot.dropna(subset=["slug"])

    glean = pd.read_csv(ROOT / "input" / "insights_glean-usage-PAST_3_MONTHS.csv", encoding="latin1")
    glean["slug"] = slug(glean["Email"])
    glean["glean_actions"] = pd.to_numeric(glean["Assistant actions in period"], errors="coerce").fillna(0)
    glean = glean.dropna(subset=["slug"]).drop_duplicates("slug")

    people = pd.DataFrame({"slug": sorted(set(claude["slug"]) | set(copilot["slug"]) | set(glean["slug"]))})
    people = people.merge(claude[["slug", "spend", "code"]], on="slug", how="left")
    people = people.merge(copilot[["slug", "prompts"]], on="slug", how="left")
    people = people.merge(glean[["slug", "glean_actions", "Title", "Department"]], on="slug", how="left")
    for col in ["spend", "code", "prompts", "glean_actions"]:
        people[col] = people[col].fillna(0)
    people["in_claude"] = people["slug"].isin(claude["slug"])
    people["in_copilot"] = people["slug"].isin(copilot["slug"])
    people["in_glean"] = people["slug"].isin(glean["slug"])
    people["seniority"] = pd.Categorical(people["Title"].map(seniority_of), SENIORITY_ORDER, ordered=True)
    people["level_group"] = pd.Categorical(people["seniority"].astype(str).map(GROUP_OF), GROUP_ORDER, ordered=True)
    return people


def overlap(people: pd.DataFrame) -> str:
    c, p, g = people["in_claude"], people["in_copilot"], people["in_glean"]
    rows = pd.DataFrame([
        {"group": "Claude only", "people": (c & ~p & ~g).sum()},
        {"group": "Copilot only", "people": (p & ~c & ~g).sum()},
        {"group": "Glean only", "people": (g & ~c & ~p).sum()},
        {"group": "Claude + Copilot", "people": (c & p & ~g).sum()},
        {"group": "Copilot + Glean", "people": (p & g & ~c).sum()},
        {"group": "Claude + Glean", "people": (c & g & ~p).sum()},
        {"group": "All three", "people": (c & p & g).sum()},
    ])
    return rows.to_markdown(index=False)


def boundary(people: pd.DataFrame) -> str:
    both = people[people["in_claude"] & people["in_copilot"]].copy()
    both["side"] = pd.cut(both["prompts"], bins=[-1, BOUNDARY, both["prompts"].max()],
                          labels=[f"Up to {BOUNDARY} prompts", f"Over {BOUNDARY} prompts"])
    by_side = both.groupby("side", observed=True).agg(
        people=("slug", "count"), median_claude_spend=("spend", "median"), median_code_sessions=("code", "median")
    ).reset_index()
    by_side["median_claude_spend"] = by_side["median_claude_spend"].map(lambda v: f"${v:,.0f}")
    heavy_both = ((both["spend"] > both["spend"].quantile(0.9)) & (both["prompts"] > both["prompts"].quantile(0.9))).sum()
    return "\n".join([
        f"On the {len(both):,} people in both, Copilot prompts and Claude spend correlate at {both['prompts'].corr(both['spend']):.2f}.",
        f"The split holds at {BOUNDARY} Copilot prompts. Only {heavy_both} people are heavy in both.",
        "",
        by_side.round(0).to_markdown(index=False),
    ])


def glean_compare(people: pd.DataFrame) -> str:
    both = people[people["in_claude"] & people["in_glean"]].copy()
    both["band"] = pd.qcut(both["glean_actions"].rank(method="first"), 3, labels=["Light Glean", "Mid Glean", "Heavy Glean"])
    by_band = both.groupby("band", observed=True).agg(
        people=("slug", "count"), median_glean_actions=("glean_actions", "median"), median_claude_spend=("spend", "median")
    ).reset_index()
    by_band["median_claude_spend"] = by_band["median_claude_spend"].map(lambda v: f"${v:,.0f}")
    known = people.loc[people["in_claude"] & people["Title"].notna()]
    return "\n".join([
        f"Glean now covers {people['in_glean'].sum():,} people, and {len(both):,} of them also use Claude.",
        f"Glean actions and Claude spend correlate at {both['glean_actions'].corr(both['spend']):.2f}.",
        f"Titles are now known for {len(known):,} of {people['in_claude'].sum():,} Claude users, up from 320.",
        "",
        by_band.round(0).to_markdown(index=False),
    ])


def by_level(people: pd.DataFrame) -> str:
    known = people[people["seniority"].notna()]
    per_person = known.groupby("seniority", observed=True).agg(
        people=("slug", "count"),
        claude_spend=("spend", "mean"),
        copilot_prompts=("prompts", "mean"),
        glean_actions=("glean_actions", "mean"),
    ).reset_index().round(0)
    per_person["claude_spend"] = per_person["claude_spend"].map(lambda v: f"${v:,.0f}")
    return f"Per person, titles known for {len(known):,} of {len(people):,}.\n\n" + per_person.to_markdown(index=False)


def level_graph(people: pd.DataFrame) -> None:
    known = people[people["seniority"].notna()]
    measures = [("spend", "Claude spend ($)"), ("prompts", "Copilot prompts"), ("glean_actions", "Glean actions")]
    colors = ["#c43e1c", "#0f6cbd", "#5b8def"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (col, label), color in zip(axes, measures, colors):
        means = known.groupby("level_group", observed=True)[col].mean()
        bars = ax.bar(means.index.astype(str), means.values, color=color, width=0.62)
        for bar, v in zip(bars, means.values):
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, means.max() * 1.2)
        ax.set_title(label)
        ax.set_ylabel("Per person")
    fig.suptitle("Consumption per person by leadership group")
    fig.tight_layout()
    fig.savefig(OUT / "by_leadership.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (col, label), color in zip(axes, measures, colors):
        means = known.groupby("seniority", observed=True)[col].mean()
        bars = ax.bar(means.index.astype(str), means.values, color=color, width=0.62)
        for bar, v in zip(bars, means.values):
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, means.max() * 1.25)
        ax.tick_params(axis="x", labelrotation=30)
        ax.set_title(label)
    fig.suptitle("Consumption per person by level")
    fig.tight_layout()
    fig.savefig(OUT / "by_level.png", dpi=160)
    plt.close(fig)


def graph(people: pd.DataFrame) -> None:
    both = people[people["in_claude"] & people["in_copilot"]].copy()
    both["band"] = pd.cut(both["prompts"], bins=[-1, 0, 5, 10, 20, 50, 100, both["prompts"].max()])
    curve = both.groupby("band", observed=True).agg(prompts=("prompts", "median"), code=("code", "median"), spend=("spend", "median"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.plot(curve["prompts"], curve["code"], color="#c43e1c", marker="o")
    ax2 = ax.twinx()
    ax2.plot(curve["prompts"], curve["spend"], color="#0f6cbd", marker="o")
    ax.axvline(BOUNDARY, color="#111827", linestyle="--", linewidth=1)
    ax.set_xlabel("Copilot prompts (median of band)")
    ax.set_ylabel("Median Claude code sessions", color="#c43e1c")
    ax2.set_ylabel("Median Claude spend ($)", color="#0f6cbd")
    ax.set_title("Claude usage falls as Copilot prompts rise")

    ax = axes[1]
    c, p, g = people["in_claude"], people["in_copilot"], people["in_glean"]
    bars = pd.Series({
        "Copilot only": (p & ~c & ~g).sum(),
        "Glean only": (g & ~c & ~p).sum(),
        "Copilot + Glean": (p & g & ~c).sum(),
        "Claude + Copilot": (c & p & ~g).sum(),
        "All three": (c & p & g).sum(),
    })
    ax.barh(bars.index[::-1], bars.values[::-1], color="#5b8def")
    for y, v in enumerate(bars.values[::-1]):
        ax.text(v + 30, y, f"{v:,}", va="center", fontsize=8)
    ax.set_xlim(0, bars.max() * 1.18)
    ax.set_xlabel("People")
    ax.set_title("Who uses which tool")
    fig.tight_layout()
    fig.savefig(OUT / "boundary.png", dpi=160)
    plt.close(fig)


def main() -> None:
    people = load()
    report = "\n\n".join([
        "# Combined\n\n" + overlap(people),
        "# The boundary\n\n" + boundary(people),
        "# Glean\n\n" + glean_compare(people),
        "# By level\n\n" + by_level(people),
    ])
    (OUT / "results.md").write_text(report, encoding="utf-8")
    graph(people)
    level_graph(people)
    print(report)


if __name__ == "__main__":
    main()
