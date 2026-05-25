"""Tools exposed to the data analyst agent.

Each tool has a clear name, a description aimed at an LLM picking between
tools, and a Pydantic input schema. The descriptions intentionally spell out
*when* a tool should be used, since that signal is what the router/agent
relies on.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from . import dataset as ds

# Cap to keep tool outputs LLM-friendly. Tools should fail closed rather than
# dump 27k rows into the context window.
MAX_EXAMPLES = 25


# -- Pydantic input schemas ---------------------------------------------------


class _Empty(BaseModel):
    """No arguments."""


class ListIntentsInput(BaseModel):
    """Input for ``list_intents``."""

    category: Optional[str] = Field(
        default=None,
        description=(
            "Optional category name to scope the result to (case-insensitive, "
            "e.g. 'ACCOUNT'). If omitted, returns intents across the whole "
            "dataset."
        ),
    )


class CountRecordsInput(BaseModel):
    """Input for ``count_records``."""

    category: Optional[str] = Field(
        default=None,
        description="Optional category to filter by (case-insensitive, e.g. 'REFUND').",
    )
    intent: Optional[str] = Field(
        default=None,
        description=(
            "Optional intent to filter by (case-insensitive, e.g. 'get_refund'). "
            "Combine with `category` for narrower counts."
        ),
    )


class GetExamplesInput(BaseModel):
    """Input for ``get_examples``."""

    category: Optional[str] = Field(
        default=None,
        description="Optional category to filter by (case-insensitive).",
    )
    intent: Optional[str] = Field(
        default=None,
        description="Optional intent to filter by (case-insensitive).",
    )
    n: int = Field(
        default=5,
        ge=1,
        le=MAX_EXAMPLES,
        description=f"How many example rows to return (1-{MAX_EXAMPLES}).",
    )


class SearchExamplesInput(BaseModel):
    """Input for ``search_examples``."""

    query: str = Field(
        description=(
            "Free-text phrase to search for inside customer instructions. "
            "Use this when the user describes a request paraphrastically "
            "(e.g. 'people wanting their money back') rather than naming an "
            "intent directly."
        ),
    )
    n: int = Field(
        default=5,
        ge=1,
        le=MAX_EXAMPLES,
        description=f"How many matching example rows to return (1-{MAX_EXAMPLES}).",
    )


class IntentDistributionInput(BaseModel):
    """Input for ``intent_distribution``."""

    category: Optional[str] = Field(
        default=None,
        description=(
            "Optional category to scope the distribution to (case-insensitive). "
            "If omitted, returns counts across the whole dataset."
        ),
    )


# -- Tools --------------------------------------------------------------------


@tool("list_categories", args_schema=_Empty)
def list_categories() -> list[str]:
    """List every distinct category in the Bitext customer-service dataset.

    Use this as the first step whenever the user asks what categories exist,
    or when you need to map a vague topic (e.g. 'shipping problems') to a
    real category name.
    """
    return ds.categories()


@tool("list_intents", args_schema=ListIntentsInput)
def list_intents(category: Optional[str] = None) -> list[str]:
    """List every distinct intent, optionally scoped to one category.

    Use this when you need to discover the exact intent name to filter on
    (e.g. before counting refunds you may want to confirm the intent is
    spelled ``get_refund``).
    """
    return ds.intents(category)


@tool("count_records", args_schema=CountRecordsInput)
def count_records(
    category: Optional[str] = None, intent: Optional[str] = None
) -> dict[str, object]:
    """Count rows in the dataset, optionally filtered by category and/or intent.

    Use this for "how many ..." questions. Returns a dict with the resulting
    ``count`` and the filters that were applied so you can echo them back
    to the user.
    """
    df = ds.load_dataset_df()
    applied: dict[str, str] = {}
    if category is not None:
        df = df[df["category"].str.upper() == category.upper()]
        applied["category"] = category.upper()
    if intent is not None:
        df = df[df["intent"].str.lower() == intent.lower()]
        applied["intent"] = intent.lower()
    return {"count": int(len(df)), "filters": applied}


def _rows_to_records(df, n: int) -> list[dict[str, str]]:
    take = df.head(n)
    return [
        {
            "category": str(row["category"]),
            "intent": str(row["intent"]),
            "instruction": str(row["instruction"]),
            "response": str(row["response"]),
        }
        for _, row in take.iterrows()
    ]


@tool("get_examples", args_schema=GetExamplesInput)
def get_examples(
    category: Optional[str] = None,
    intent: Optional[str] = None,
    n: int = 5,
) -> list[dict[str, str]]:
    """Fetch example rows from the dataset, optionally filtered.

    Each example contains ``category``, ``intent``, ``instruction`` (the
    customer message) and ``response`` (the agent reply). Use this when the
    user asks to *see* examples, OR when you need raw text to summarize an
    open-ended question (e.g. "summarize the FEEDBACK category" — call this
    with ``category='FEEDBACK'`` and a larger ``n``).
    """
    df = ds.load_dataset_df()
    if category is not None:
        df = df[df["category"].str.upper() == category.upper()]
    if intent is not None:
        df = df[df["intent"].str.lower() == intent.lower()]
    return _rows_to_records(df, n)


@tool("search_examples", args_schema=SearchExamplesInput)
def search_examples(query: str, n: int = 5) -> list[dict[str, str]]:
    """Find example rows whose customer instruction matches a free-text phrase.

    Performs a simple case-insensitive substring search on the
    ``instruction`` column. Use this when the user paraphrases what they
    want (e.g. "people wanting their money back", "complaints about late
    delivery") instead of naming a category or intent.
    """
    df = ds.load_dataset_df()
    mask = df["instruction"].str.contains(query, case=False, na=False, regex=False)
    return _rows_to_records(df[mask], n)


@tool("intent_distribution", args_schema=IntentDistributionInput)
def intent_distribution(category: Optional[str] = None) -> dict[str, int]:
    """Return a count of rows per intent, optionally scoped to one category.

    Use this for "distribution of ..." questions. The returned dict is
    sorted by descending count.
    """
    df = ds.load_dataset_df()
    if category is not None:
        df = df[df["category"].str.upper() == category.upper()]
    counts = df["intent"].value_counts()
    return {str(k): int(v) for k, v in counts.items()}


ALL_TOOLS = [
    list_categories,
    list_intents,
    count_records,
    get_examples,
    search_examples,
    intent_distribution,
]
