"""Query router: classify an incoming user question.

The router decides which branch of the graph handles a question, so the
agent doesn't waste tool calls on something it can't answer (out-of-scope)
or use the wrong reasoning style for an open-ended question.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from .llm import build_chat_model

QueryKind = Literal["structured", "unstructured", "out_of_scope"]


class RouteDecision(BaseModel):
    """Structured output emitted by the router LLM."""

    kind: QueryKind = Field(
        description=(
            "Classification of the user's question. "
            "'structured' = concrete, data-driven answer derivable from filters/counts/examples. "
            "'unstructured' = open-ended summarization or qualitative description that requires "
            "reading raw text. "
            "'out_of_scope' = unrelated to the Bitext customer-service dataset."
        )
    )
    reason: str = Field(
        description="One short sentence justifying the classification (for logging)."
    )


_SYSTEM_PROMPT = """You are the router for a data-analyst agent.

The agent answers questions about the **Bitext Customer Service Tagged
Training Dataset** -- a table of synthetic customer-support messages with
columns: category (e.g. ACCOUNT, ORDER, REFUND, SHIPPING, FEEDBACK, ...),
intent (e.g. get_refund, cancel_order, ...), instruction (the customer
message) and response (the agent reply).

Classify the user's question into exactly one of three buckets:

- "structured": the answer is concrete data that can be computed by
  filtering / counting / listing examples. Examples:
  * "How many refund requests did we get?"
  * "What categories exist?"
  * "Show me 3 examples from SHIPPING."
  * "What is the distribution of intents in the ACCOUNT category?"

- "unstructured": the answer requires reading and summarizing free-text
  rows from the dataset. Examples:
  * "Summarize the FEEDBACK category."
  * "How do agents typically respond to cancellation requests?"

- "out_of_scope": the question is unrelated to this dataset and would
  require general world knowledge, creative writing, or external lookup.
  Examples:
  * "Who won the 2024 Champions League?"
  * "Write me a poem about customer service."
  * "What's the best CRM software?"

Be strict about out_of_scope: if answering would require knowledge that is
NOT in the customer-service dataset, classify it as out_of_scope.

ALSO IN-SCOPE (classify as "structured"): meta-questions about the
conversation itself or what the agent remembers about the user. These
are answerable from the agent's stored profile and conversation history.
Examples:
  * "What do you remember about me?"
  * "What have we talked about?"
  * "What do I usually ask about?"

Follow-up references to earlier turns (e.g. "show me 3 more", "what
about refunds?", "total of the last two") should be classified the same
way the prior question was -- usually "structured".
"""


def build_router():
    """Return a callable that classifies a user question."""
    model = build_chat_model(temperature=0.0)
    structured = model.with_structured_output(RouteDecision)
    prompt = ChatPromptTemplate.from_messages(
        [("system", _SYSTEM_PROMPT), ("human", "{question}")]
    )
    return prompt | structured
