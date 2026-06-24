"""
data_loading.py
--------------
Downloads and loads the Anthropic Economic Index dataset directly
from Hugging Face via HTTP (no datasets library required).
"""
import os
import requests
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# ── Direct CSV URLs (release_2026_03_24, Claude.ai split) ─────────────────────
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

    print(f"[data_loading] Downloading AEI data from Hugging Face...")
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


def build_task_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot the raw long-format AEI data into a cluster-level panel.
    Each row = one O*NET task cluster; columns = key statistics.

    Expected raw columns (based on release_2026_03_24 schema):
        facet, variable, value, cluster_name, level, geo_id
    Falls back to direct-column format if pre-aggregated per task.
    """
    # ── Case 1: pre-aggregated (task-row format, as in release_2026_03_24) ──
    if "task" in df.columns and "human_time_mean" in df.columns:
        panel = df.rename(columns={"task": "cluster_name"}).copy()
        return panel

    # ── Case 2: long-format facet/variable/value ──────────────────────────────
    def _extract(facet_kw: str, var_kw: str, col: str) -> pd.DataFrame:
        mask = (
            df["facet"].str.contains(facet_kw, na=False)
            & df["variable"].str.contains(var_kw, na=False)
        )
        sub = df[mask]
        # prefer global aggregate if geo_id present
        if "geo_id" in sub.columns:
            sub = sub[sub["geo_id"].isin(["OWID_WRL", "WLD", "WORLD"])]
        return (
            sub[["cluster_name", "value"]]
            .drop_duplicates("cluster_name")
            .rename(columns={"value": col})
        )

    panel = (
        _extract("human_only_time",      "mean",  "human_time_mean")
        .merge(_extract("task_success",  "pct",   "success_pct"),      on="cluster_name", how="inner")
        .merge(_extract("ai_autonomy",   "mean",  "autonomy_mean"),     on="cluster_name", how="left")
        .merge(_extract("human_with_ai_time|human_ai_time", "mean", "ai_time_mean"), on="cluster_name", how="left")
        .merge(_extract("human_education", "mean", "edu_years_mean"),   on="cluster_name", how="left")
    )
    return panel
