"""
data_loading.py
--------------
Downloads and loads the Anthropic Economic Index dataset directly
from Hugging Face via HTTP (no datasets library required), and
constructs a task-cluster-level panel used throughout the analysis.
"""
import requests
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# ── Direct CSV URL (release_2026_03_24, Claude.ai split) ─────────────────────
AEI_URL = (
    "https://huggingface.co/datasets/Anthropic/EconomicIndex/resolve/main/"
    "release_2026_03_24/data/aei_raw_claude_ai_2026-02-05_to_2026-02-12.csv"
)
AEI_FILENAME = "aei_raw_claude_ai.csv"


def download_aei(data_dir: str = "data", force: bool = False) -> Path:
    """Download AEI CSV if not already cached. Returns local path."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    dest = data_dir / AEI_FILENAME

    if dest.exists() and not force:
        print(f"[data_loading] Using cached file: {dest}")
        return dest

    print("[data_loading] Downloading AEI data from Hugging Face...")
    response = requests.get(AEI_URL, stream=True, timeout=120)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=AEI_FILENAME
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    print(f"[data_loading] Saved to {dest}")
    return dest


def load_raw(data_dir: str = "data") -> pd.DataFrame:
    """Load raw AEI CSV. Auto-downloads if missing."""
    path = download_aei(data_dir)
    df = pd.read_csv(path, low_memory=False)
    assert len(df) > 0, "Loaded dataframe is empty."
    return df


# ── onet_task facet specification ────────────────────────────────────────────
# Each entry maps an output panel column to its (facet, variable) source in the
# long-format AEI data. All are extracted at geo_id == 'GLOBAL'. `scale` is
# applied to the raw value (e.g. percent → ratio).
#
# Cluster names carry a trailing "::<suffix>". Most facets use "::value" (one
# row per task). task_success instead encodes a categorical distribution with
# "::yes" / "::no" / "::not_classified" rows; the success rate is the "::yes"
# percentage. We therefore join all facets on the suffix-stripped `task_key`
# and, when `category` is set, keep only rows whose suffix matches it.
FACET_SPECS = {
    "human_time_mean": {
        "facet":    "onet_task::human_only_time",
        "variable": "onet_task_human_only_time_mean",
        "scale":    1.0,   # hours
        "category": None,
    },
    "success_pct": {
        "facet":    "onet_task::task_success",
        "variable": "onet_task_task_success_pct",
        "scale":    0.01,  # percent → ratio in [0, 1]
        "category": "yes", # success rate = share of "::yes" outcomes
    },
    "autonomy_mean": {
        "facet":    "onet_task::ai_autonomy",
        "variable": "onet_task_ai_autonomy_mean",
        "scale":    1.0,   # 1–5 score
        "category": None,
    },
    "ai_time_mean": {
        "facet":    "onet_task::human_with_ai_time",
        "variable": "onet_task_human_with_ai_time_mean",
        "scale":    1.0,   # minutes
        "category": None,
    },
    "edu_years_mean": {
        "facet":    "onet_task::human_education_years",
        "variable": "onet_task_human_education_years_mean",
        "scale":    1.0,   # years
        "category": None,
    },
}


def _task_key(cluster_name: pd.Series) -> pd.Series:
    """Strip the trailing '::<suffix>' so facets with different suffixes join."""
    return cluster_name.str.rsplit("::", n=1).str[0]


def _extract_facet(df: pd.DataFrame, facet: str, variable: str, scale: float,
                   col: str, category: str | None = None) -> pd.DataFrame:
    """Extract one GLOBAL onet_task facet keyed by suffix-stripped task name."""
    mask = (
        (df["facet"] == facet)
        & (df["variable"] == variable)
        & (df["geo_id"] == "GLOBAL")
    )
    sub = df.loc[mask].copy()
    if category is not None:
        sub = sub[sub["cluster_name"].str.rsplit("::", n=1).str[-1] == category]
    sub["task_key"] = _task_key(sub["cluster_name"])
    sub["value"] = pd.to_numeric(sub["value"], errors="coerce") * scale
    return (
        sub.groupby("task_key", dropna=True)["value"]
        .mean()
        .reset_index(name=col)
    )


def build_task_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Construct a task-cluster-level panel from the long-format AEI data.

    For release_2026_03_24, each metric lives in its own onet_task::* facet at
    geo_id == 'GLOBAL'. Cluster names share a common task description but carry
    facet-specific suffixes, so we join on the suffix-stripped `task_key`. Every
    unique task is treated as a cluster.

    Columns produced:
        human_time_mean  hours      (onet_task::human_only_time)
        success_pct      ratio 0–1  (onet_task::task_success "::yes" share)
        autonomy_mean    1–5 score  (onet_task::ai_autonomy)
        ai_time_mean     minutes    (onet_task::human_with_ai_time)
        edu_years_mean   years      (onet_task::human_education_years)
    """
    # Base panel: human-only time defines the universe of task clusters.
    base_spec = FACET_SPECS["human_time_mean"]
    panel = _extract_facet(
        df, base_spec["facet"], base_spec["variable"],
        base_spec["scale"], "human_time_mean", base_spec["category"],
    )

    # Merge the remaining facets onto the base by task_key.
    for col, spec in FACET_SPECS.items():
        if col == "human_time_mean":
            continue
        feat = _extract_facet(df, spec["facet"], spec["variable"],
                              spec["scale"], col, spec["category"])
        panel = panel.merge(feat, on="task_key", how="left")
        n_present = panel[col].notna().sum()
        print(f"[data_loading]   {col}: {n_present}/{len(panel)} clusters populated")

    # Preserve a readable cluster_name for downstream code/plots.
    panel = panel.rename(columns={"task_key": "cluster_name"})

    print(f"[data_loading] Panel built with {len(panel)} task clusters")

    return panel
