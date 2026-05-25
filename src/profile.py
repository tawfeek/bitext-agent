"""Per-user persistent profile (semantic memory).

The profile is **not** a replay of past messages — it is a small set of
distilled facts about the user (name, recurring topics of interest,
stated preferences). It lives in its own markdown file per user under
``data/profiles/<user_id>.md`` so it survives restarts independently of
the conversation checkpoints.

The updater runs after each agent turn and is instructed to return the
profile unchanged when nothing new emerged, so the file only mutates
when there is genuinely new information.
"""

from __future__ import annotations

import re
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

from .config import SETTINGS
from .llm import build_chat_model

_EMPTY_PROFILE = (
    "# User Profile\n\n"
    "**Name:** —\n"
    "**Interests:** —\n"
    "**Preferences:** —\n"
    "**Notes:** —\n"
)

# Allow letters, digits, dot, underscore, dash. Anything else gets stripped.
_SAFE_USER_RE = re.compile(r"[^A-Za-z0-9._-]")


def _safe_user_id(user_id: str) -> str:
    cleaned = _SAFE_USER_RE.sub("_", user_id).strip("_")
    return cleaned or "default"


def profile_path(user_id: str) -> Path:
    """Return the path to a given user's profile file."""
    return SETTINGS.profiles_dir / f"{_safe_user_id(user_id)}.md"


def load_profile(user_id: str) -> str:
    """Return the markdown profile for ``user_id``, creating a blank one if missing."""
    path = profile_path(user_id)
    if not path.exists():
        path.write_text(_EMPTY_PROFILE, encoding="utf-8")
    return path.read_text(encoding="utf-8")


def save_profile(user_id: str, content: str) -> None:
    """Write the markdown profile for ``user_id`` to disk."""
    profile_path(user_id).write_text(content.strip() + "\n", encoding="utf-8")


_UPDATER_SYSTEM = """You maintain a structured profile of a single user
across many conversations with a customer-service data-analyst agent.

Below is the CURRENT profile (markdown). Then the latest user message and
the latest agent reply. Emit an UPDATED profile in markdown.

Rules:
- If nothing new about the user emerged, return the current profile
  EXACTLY UNCHANGED (same text, no rewording).
- Only record durable facts about the user: their name, recurring topics
  of interest, stated preferences, recurring goals. Do NOT replay past
  messages or store one-off question contents.
- Keep the profile concise (under 200 words total).
- Preserve the markdown structure with these bold labels in this order:
  **Name:**, **Interests:**, **Preferences:**, **Notes:**.
- If a field's value is unknown, write `—` (em dash). Never omit a field.
- Return ONLY the markdown profile. No preamble, no explanation, no
  code fences.
"""

_UPDATER_HUMAN = """CURRENT PROFILE:
---
{current_profile}
---

LATEST USER MESSAGE:
{user_message}

LATEST AGENT REPLY:
{agent_response}
"""


def build_profile_updater():
    """Return a chain that takes profile/user/agent text and yields markdown."""
    model = build_chat_model(temperature=0.0)
    prompt = ChatPromptTemplate.from_messages(
        [("system", _UPDATER_SYSTEM), ("human", _UPDATER_HUMAN)]
    )
    return prompt | model


__all__ = [
    "load_profile",
    "save_profile",
    "profile_path",
    "build_profile_updater",
]
