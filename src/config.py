"""Runtime configuration loaded from environment / .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


@dataclass(frozen=True)
class Settings:
    """Resolved settings for the agent."""

    nebius_api_key: str
    nebius_base_url: str
    nebius_model: str
    max_iterations: int
    dataset_cache_path: Path
    hf_dataset_id: str

    @classmethod
    def load(cls) -> "Settings":
        return cls(
            nebius_api_key=os.environ.get("NEBIUS_API_KEY", ""),
            nebius_base_url=os.environ.get(
                "NEBIUS_BASE_URL", "https://api.studio.nebius.com/v1/"
            ),
            nebius_model=os.environ.get("NEBIUS_MODEL", "Qwen/Qwen3-235B-A22B"),
            max_iterations=int(os.environ.get("AGENT_MAX_ITERATIONS", "12")),
            dataset_cache_path=DATA_DIR / "bitext.parquet",
            hf_dataset_id="bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        )


SETTINGS = Settings.load()
