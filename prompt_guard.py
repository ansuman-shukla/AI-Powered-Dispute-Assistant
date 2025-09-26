"""Rule-based prompt safety guard for the dispute analysis assistant."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Pattern, Tuple


@dataclass(frozen=True)
class Rule:  # pragma: no cover - simple data holder
    """Encapsulates a safety rule backed by a regular expression pattern."""

    name: str
    pattern: Pattern[str]
    message: str


class PromptSafetyChecker:
    """Rule-based checker to stop malicious, toxic, or off-topic prompts."""

    _ALLOWED_DOMAIN_KEYWORDS: Tuple[str, ...] = (
        "dispute",
        "transaction",
        "charge",
        "refund",
        "customer",
        "merchant",
        "payment",
        "analysis",
        "dashboard",
        "report",
        "csv",
        "data",
        "bank",
        "model",
    )

    def __init__(self) -> None:
        self._rules: List[Rule] = [
            Rule(
                name="prompt_attack",
                pattern=re.compile(
                    r"(?i)(?:ignore|disregard|forget)\b[\s\S]{0,80}\b(?:instruction|previous|earlier|system|rules?)",
                ),
                message=(
                    "The request attempts to override or forget the assistant's safety instructions."
                ),
            ),
            Rule(
                name="prompt_attack",
                pattern=re.compile(
                    r"(?i)(?:reveal|show|display|print)\b[\s\S]{0,80}\b(?:system prompt|hidden instruction|policy|guardrail)",
                ),
                message="The request is trying to expose confidential system prompts or policies.",
            ),
            Rule(
                name="prompt_attack",
                pattern=re.compile(
                    r"(?i)(?:bypass|override|disable)\b[\s\S]{0,80}\b(?:safety|guardrail|filter|constraint|content policy)",
                ),
                message="The request is attempting to disable guardrails or policies.",
            ),
            Rule(
                name="prompt_attack",
                pattern=re.compile(
                    r"(?i)(?:act as|pretend to be|you are now)\b[\s\S]{0,80}\b(?:root|superuser|system|hacker|developer mode)",
                ),
                message="Role-play request tries to escalate privileges beyond the allowed assistant scope.",
            ),
            Rule(
                name="toxicity",
                pattern=re.compile(
                    r"(?i)\b(?:fuck|shit|bitch|asshole|bastard|retard|moron|slur)\b",
                ),
                message="The prompt includes abusive or profane language.",
            ),
            Rule(
                name="toxicity",
                pattern=re.compile(
                    r"(?i)(?:kill|murder|hurt|harm|destroy)\b[\s\S]{0,40}\b(?:customer|agent|people|someone)",
                ),
                message="The prompt contains violent intent toward individuals.",
            ),
        ]

        self._off_topic_patterns: List[Pattern[str]] = [
            re.compile(r"(?i)\b(?:politics?|election|government policy)\b"),
            re.compile(r"(?i)\b(?:celebrity|gossip|movie|music chart)\b"),
            re.compile(r"(?i)\b(?:weapon|bomb|explosive|terror|hack)\b"),
            re.compile(r"(?i)\b(?:recipe|cook|bake|cuisine)\b"),
            re.compile(r"(?i)\b(?:adult content|porn|sexual)\b"),
        ]

    def check_prompt(self, prompt: str) -> Tuple[bool, Dict[str, str]]:
        """
        Evaluate a prompt and return (is_safe, metadata dict).

        The metadata always contains:
          * ``category`` - the rule category triggered (or "safe")
          * ``message`` - a short human-readable description
          * ``matched`` - the raw pattern or keyword that fired (for auditing)
        """

        normalized = " ".join((prompt or "").lower().split())

        for rule in self._rules:
            if rule.pattern.search(normalized):
                return False, {
                    "category": rule.name,
                    "message": rule.message,
                    "matched": rule.pattern.pattern,
                }

        if self._is_off_topic(normalized):
            return False, {
                "category": "topic_veering",
                "message": "The request is off-topic relative to dispute analysis.",
                "matched": "off_topic_keyword",
            }

        return True, {
            "category": "safe",
            "message": "Prompt is within the allowed scope.",
            "matched": "",
        }

    def _is_off_topic(self, normalized_prompt: str) -> bool:
        """Detect prompts that drift from the business/analytics focus."""

        if any(keyword in normalized_prompt for keyword in self._ALLOWED_DOMAIN_KEYWORDS):
            return False

        return any(pattern.search(normalized_prompt) for pattern in self._off_topic_patterns)


# Shared singleton instance
DEFAULT_PROMPT_SAFETY_CHECKER = PromptSafetyChecker()
