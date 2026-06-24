"""
data_loading.py
--------------
Downloads and loads the Anthropic Economic Index dataset directly
from Hugging Face via HTTP (no datasets library required).
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


def build_task_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot AEI long-format data into a cluster-level panel.

    Each row = one O*NET task cluster (cluster_name), with columns:
        - human_time_mean
        - success_pct
        - autonomy_mean
        - ai_time_mean
        - edu_years_mean

    This function is tailored to the release_2026_03_24 schema, which uses
    facet / variable / value with global aggregates marked by geo_id == 'GLOBAL'.
    """

    def _extract(facet_kw: str, var_kw: str, col: str) -> pd.DataFrame:
        # Filter by facet + variable substring
        mask = (
            df["facet"].str.contains(facet_kw, na=False)
            & df["variable"].str.contains(var_kw, na=False)
        )
        sub = df[mask].copy()

        # Prefer global aggregate if geo_id present
        if "geo_id" in sub.columns:
            global_codes = ["GLOBAL", "OWID_WRL", "WLD", "WORLD"]
            sub_global = sub[sub["geo_id"].isin(global_codes)]
            if len(sub_global) > 0:
                sub = sub_global

        # Drop rows without a task cluster_name (these are pure global stats)
        if "cluster_name" in sub.columns:
            sub = sub.dropna(subset=["cluster_name"])

        return (
            sub[["cluster_name", "value"]]
            .drop_duplicates("cluster_name")
            .rename(columns={"value": col})
        )

    # Cluster-level human-only time (hours)
    time_mean = _extract("onet_task::human_only_time", "mean", "human_time_mean")

    # Task success rate (%)
    success = _extract("onet_task::task_success", "pct", "success_pct")

    # AI autonomy (1–5)
    autonomy = _extract("onet_task::ai_autonomy", "mean", "autonomy_mean")

    # Human-with-AI time (minutes)
    ai_time = _extract("onet_task::human_with_ai_time", "mean", "ai_time_mean")

    # Human education years
    edu = _extract("onet_task::human_education", "mean", "edu_years_mean")

    panel = (
        time_mean
        .merge(success, on="cluster_name", how="inner")
        .merge(autonomy, on="cluster_name", how="left")
        .merge(ai_time, on="cluster_name", how="left")
        .merge(edu, on="cluster_name", how="left")
    )

    if panel["human_time_mean"].nunique() < 3:
        raise ValueError("Not enough variation in human_time_mean across task clusters.")

    return panel
