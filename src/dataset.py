"""Loader for the Bitext Customer Service Tagged Training dataset.

The dataset is fetched once from HuggingFace and cached locally as parquet
so that subsequent runs are offline-friendly. The resulting DataFrame has the
columns ``flags``, ``instruction``, ``category``, ``intent``, ``response``.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from .config import SETTINGS


def _download_from_hf() -> pd.DataFrame:
    """Download the dataset from HuggingFace and return it as a DataFrame."""
    from datasets import load_dataset

    ds = load_dataset(SETTINGS.hf_dataset_id, split="train")
    return ds.to_pandas()


@lru_cache(maxsize=1)
def load_dataset_df() -> pd.DataFrame:
    """Return the Bitext dataset as a pandas DataFrame, using a local cache.

    The first call downloads from HuggingFace and writes a parquet file under
    ``data/``. Later calls (in the same process or new ones) read from that
    parquet directly.
    """
    cache_path = SETTINGS.dataset_cache_path
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
    else:
        df = _download_from_hf()
        df.to_parquet(cache_path, index=False)

    # Normalize column names just in case the upstream schema shifts.
    df.columns = [c.lower() for c in df.columns]
    return df


def categories() -> list[str]:
    """Return the sorted list of unique categories."""
    return sorted(load_dataset_df()["category"].dropna().unique().tolist())


def intents(category: str | None = None) -> list[str]:
    """Return the sorted list of unique intents, optionally filtered by category."""
    df = load_dataset_df()
    if category is not None:
        df = df[df["category"].str.upper() == category.upper()]
    return sorted(df["intent"].dropna().unique().tolist())
