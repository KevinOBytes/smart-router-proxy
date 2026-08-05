"""Minimal deterministic checks — just tool-name detection for computer_use.

All heavy classification is handled by the BERT classifier.
"""

from __future__ import annotations

from smart_router_proxy.models import (
    ClassifierResult,
    RiskLevel,
    Sensitivity,
    TaskClass,
)


def classify_deterministic(
    user_request: str,
    tool_names: list[str] | None = None,
) -> ClassifierResult | None:
    """Only handles tool-name-based computer_use detection.

    Everything else defers to the BERT classifier.
    """
    if tool_names and any("computer" in t.lower() for t in tool_names):
        return ClassifierResult(
            task_class=TaskClass.COMPUTER_USE,
            risk=RiskLevel.MODERATE,
            sensitivity=Sensitivity.INTERNAL,
            requires_tools=True,
            requires_vision=True,
            long_context=False,
            destructive_potential=False,
            confidence=0.95,
        )
    return None
