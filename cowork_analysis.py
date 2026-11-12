"""Cowork spend, usage, concentration, and dead weight.

Seniority and division come from NB1Copilot.csv, matched on email slug.
Most cowork rows have no match, so seniority coverage is partial.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
COWORK_PATH = ROOT / "cowork_sessions.csv"
NB_PATH = ROOT / "NB1Copilot.csv"
GLEAN_PATH = ROOT / "GleanUsage.xlsx"
FAST_PATH = Path.home() / "OneDrive - Gen Digital Inc" / "FastCopilotActivityUserDetail8_4_202690_days.csv"
OUT_DIR = ROOT / "cowork_output"

SENIORITY_ORDER = ["Junior", "Mid", "Senior", "Principal", "Manager", "Director+", "Unknown"]


def slug(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    return text.split("@", 1)[0] or None


def infer_seniority(title: object) -> str:
    if pd.isna(title) or not str(title).strip():
        return "Unknown"
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


def glean() -> pd.DataFrame:
    g = pd.read_excel(GLEAN_PATH)
    g["slug"] = g["Email"].map(slug)
    g["glean_chats"] = pd.to_numeric(g["Chats in period"], errors="coerce").fillna(0)
    g["glean_days"] = pd.to_numeric(g["Days active on Assistant in period"], errors="coerce").fillna(0)
    g["glean_title"] = g["Division Title"].fillna(g["Source Title"])
    g["glean_division"] = g["L1 Division"]
    return g.dropna(subset=["slug"]).drop_duplicates("slug")


def load() -> pd.DataFrame:
    cw = pd.read_csv(COWORK_PATH)
    cw["slug"] = cw["Email"].map(slug)
    nb = pd.read_csv(NB_PATH, encoding="latin1")
    nb["slug"] = nb["Email"].map(slug)
    nb = nb.dropna(subset=["slug"]).drop_duplicates("slug")
    lookup = nb.set_index("slug")[["Title", "L1 Division"]]
    g = glean().set_index("slug")
    cw["title"] = cw["slug"].map(lookup["Title"]).fillna(cw["slug"].map(g["glean_title"]))
    cw["division"] = cw["slug"].map(lookup["L1 Division"]).fillna(cw["slug"].map(g["glean_division"])).fillna("Unknown")
    cw["seniority"] = pd.Categorical(cw["title"].map(infer_seniority), SENIORITY_ORDER, ordered=True)
    cw["glean_chats"] = cw["slug"].map(g["glean_chats"])
    cw["glean_days"] = cw["slug"].map(g["glean_days"])
    fast = pd.read_csv(FAST_PATH, encoding="latin1")
    fast["slug"] = fast["User Principal Name"].map(slug)
    fast["copilot_prompts"] = pd.to_numeric(fast["Prompts submitted for All Apps"], errors="coerce").fillna(0)
    fast = fast.dropna(subset=["slug"]).drop_duplicates("slug").set_index("slug")
    cw["copilot_prompts"] = cw["slug"].map(fast["copilot_prompts"])
    cw["spend"] = pd.to_numeric(cw["Estimated Spend (USD)"], errors="coerce").fillna(0)
    for col in ["Chats", "Cowork Sessions", "Code sessions", "File Edits", "Days Active"]:
        cw[col] = pd.to_numeric(cw[col], errors="coerce").fillna(0)
    cw["activity"] = cw[["Chats", "Cowork Sessions", "Code sessions", "File Edits"]].sum(axis=1)
    surfaces = cw[["Chats", "Cowork Sessions", "Code sessions"]].rename(
        columns={"Chats": "Chat", "Cowork Sessions": "Cowork", "Code sessions": "Code"}
    )
    cw["surface"] = surfaces.idxmax(axis=1).where(surfaces.sum(axis=1) > 0, "None")
    return cw


def money(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda v: f"${v:,.0f}")
    return out


def who_spends(df: pd.DataFrame) -> str:
    spenders = df[df["spend"] > 0].sort_values("spend", ascending=False)
    total = df["spend"].sum()
    known = int(df["title"].notna().sum())
    by_seniority = (
        spenders.groupby("seniority", observed=False)["spend"]
        .agg(people="count", total="sum", median="median").reset_index()
    )
    by_seniority["share"] = (by_seniority["total"] / total).map(lambda v: f"{v:.0%}")
    by_division = (
        spenders.groupby("division")["spend"]
        .agg(people="count", total="sum").sort_values("total", ascending=False).reset_index()
    )
    by_division["share"] = (by_division["total"] / total).map(lambda v: f"{v:.0%}")
    top = spenders[["Name", "seniority", "division", "spend", "surface"]].head(15).reset_index(drop=True)
    top.index = top.index + 1
    lines = [
        f"{len(df):,} people, {len(spenders):,} of them spent, ${total:,.0f} in total.",
        f"Seniority is known for {known:,} of {len(df):,} people, so most spend falls under Unknown.",
        "",
        "## Spend by seniority",
        "",
        money(by_seniority.round(0)).to_markdown(index=False),
        "",
        "## Spend by division",
        "",
        money(by_division.round(0)).to_markdown(index=False),
        "",
        "## Top 15 spenders",
        "",
        money(top).to_markdown(),
    ]
    spenders[["slug", "Name", "seniority", "division", "title", "spend", "surface"]].to_csv(
        OUT_DIR / "spenders.csv", index=False
    )
    return "\n".join(lines)


def usage(df: pd.DataFrame) -> str:
    active = df[df["activity"] > 0]
    mix = active["surface"].value_counts().rename_axis("surface").reset_index(name="people")
    mix["share"] = (mix["people"] / len(active)).map(lambda v: f"{v:.0%}")
    by_seniority = (
        active.groupby("seniority", observed=False)[["Chats", "Cowork Sessions", "Code sessions", "Days Active"]]
        .median().round(1).reset_index()
    )
    lines = [
        f"{len(active):,} people had some activity. Each is tagged by whichever surface they used most.",
        "",
        "## Surface mix",
        "",
        mix.to_markdown(index=False),
        "",
        "## Median activity by seniority",
        "",
        by_seniority.to_markdown(index=False),
    ]
    return "\n".join(lines)


def concentration(df: pd.DataFrame) -> str:
    ranked = df.sort_values("spend", ascending=False)
    total = ranked["spend"].sum()
    rows = []
    for pct in (1, 5, 10, 20):
        n = max(1, round(len(ranked) * pct / 100))
        rows.append({"slice": f"top {pct}%", "people": n, "share of spend": f"{ranked['spend'].head(n).sum() / total:.0%}"})
    return "Spend is spread out rather than held by a handful of people.\n\n" + pd.DataFrame(rows).to_markdown(index=False)


def glean_compare(df: pd.DataFrame) -> str:
    both = df[df["glean_chats"].notna()].copy()
    both["band"] = pd.qcut(both["glean_chats"], 3, labels=["Light Glean", "Mid Glean", "Heavy Glean"])
    by_band = both.groupby("band", observed=True).agg(
        people=("slug", "count"),
        median_cowork_spend=("spend", "median"),
        median_cowork_sessions=("Cowork Sessions", "median"),
        median_code_sessions=("Code sessions", "median"),
    ).reset_index()
    corr = both["glean_chats"].corr(both["spend"])
    return "\n".join([
        f"{len(both):,} people appear in both. Glean chats and cowork spend move together only weakly (correlation {corr:.2f}).",
        "",
        by_band.round(0).to_markdown(index=False),
    ])


def copilot_compare(df: pd.DataFrame) -> str:
    both = df[df["copilot_prompts"].notna()].copy()
    both["band"] = pd.qcut(both["copilot_prompts"].rank(method="first"), 4, labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"])
    by_band = both.groupby("band", observed=True).agg(
        people=("slug", "count"),
        median_copilot_prompts=("copilot_prompts", "median"),
        median_cowork_spend=("spend", "median"),
        median_code_sessions=("Code sessions", "median"),
    ).reset_index()
    corr = both["copilot_prompts"].corr(both["spend"])
    shown = by_band.round(0)
    shown["median_cowork_spend"] = shown["median_cowork_spend"].map(lambda v: f"${v:,.0f}")
    return "\n".join([
        f"{len(both):,} of {len(df):,} cowork people match the Copilot activity export. The export has no title, so it cannot fill seniority.",
        f"Copilot prompts and cowork spend barely relate (correlation {corr:.2f}). The heaviest Copilot users spend the least on cowork.",
        "",
        shown.to_markdown(index=False),
    ])


def dead_weight(df: pd.DataFrame) -> str:
    dead = df[(df["Last Active"].isna()) & (df["activity"] == 0) & (df["spend"] == 0)]
    zero = int((df["spend"] == 0).sum())
    rows = pd.DataFrame([
        {"group": "No activity and no spend", "people": len(dead), "share": f"{len(dead) / len(df):.0%}"},
        {"group": "Zero spend", "people": zero, "share": f"{zero / len(df):.0%}"},
    ])
    dead[["slug", "Name"]].to_csv(OUT_DIR / "dead_weight.csv", index=False)
    return (
        "No activity means no last-active date and nothing in chat, cowork, or code.\n\n"
        + rows.to_markdown(index=False)
    )


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    df = load()
    report = "\n\n".join([
        "# Cowork usage and spend\n\n" + who_spends(df),
        "# Usage\n\n" + usage(df),
        "# Concentration\n\n" + concentration(df),
        "# Glean comparison\n\n" + glean_compare(df),
        "# Copilot comparison\n\n" + copilot_compare(df),
        "# Dead weight\n\n" + dead_weight(df),
    ])
    (OUT_DIR / "results.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
