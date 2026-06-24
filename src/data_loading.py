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


def build_task_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Construct a task-cluster-level panel from the long-format AEI data.

    For release_2026_03_24, cluster-level human time is given by rows with:
        facet    == 'onet_task::human_only_time'
        variable == 'onet_task_human_only_time_mean'
        geo_id   == 'GLOBAL'

    Each unique `cluster_name` is effectively a task description; we treat
    these as our task clusters.

    TODO:
        - Extend this panel to merge additional onet_task::* facets, e.g.:
          * onet_task::task_success          → success_pct
          * onet_task::ai_autonomy           → autonomy_mean
          * onet_task::human_with_ai_time    → ai_time_mean
          * onet_task::human_education_years → edu_years_mean
        - Use cluster_name and geo_id as join keys, keeping GLOBAL aggregates
          for the main analysis.
    """

    # Filter to global onet_task human-only time means
    mask = (
        (df["facet"] == "onet_task::human_only_time")
        & (df["variable"] == "onet_task_human_only_time_mean")
        & (df["geo_id"] == "GLOBAL")
    )
    sub = df[mask].copy()

    panel = (
        sub.groupby("cluster_name", dropna=True)["value"]
        .mean()
        .reset_index(name="human_time_mean")
    )

    # Placeholder columns for compatibility with downstream code. These will
    # be populated once the corresponding facets are merged as described above.
    panel["success_pct"]    = pd.NA
    panel["autonomy_mean"]  = pd.NA
    panel["ai_time_mean"]   = pd.NA
    panel["edu_years_mean"] = pd.NA

    print(f"[data_loading] Panel built with {len(panel)} task clusters")

    return panel
