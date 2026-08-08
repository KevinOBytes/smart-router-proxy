"""Domain models, enums, and contracts for the smart-router plugin."""

from __future__ import annotations

import enum

from pydantic import BaseModel, Field, field_validator

# ── Enums ──────────────────────────────────────────────────────────────


class TaskClass(enum.StrEnum):
    """The eight allowed task classifications."""

    STRUCTURED_SIMPLE = "structured_simple"
    AGENTIC_EXECUTION = "agentic_execution"
    SOFTWARE_ENGINEERING = "software_engineering"
    SECURITY_ENGINEERING = "security_engineering"
    KNOWLEDGE_REASONING = "knowledge_reasoning"
    WRITING_COMMUNICATION = "writing_communication"
    COMPUTER_USE = "computer_use"
    VISUAL_FRONTEND = "visual_frontend"


class RiskLevel(enum.StrEnum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class Sensitivity(enum.StrEnum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class OperatingMode(enum.StrEnum):
    SHADOW = "shadow"
    ACTIVE = "active"
    FIXED = "fixed"


class ReasonCode(enum.StrEnum):
    """Reason codes for route selection and escalation."""

    CLASSIFIER = "classifier"
    DETERMINISTIC = "deterministic"
    CLASSIFIER_FALLBACK = "classifier_fallback"
    ESCALATION_FAILURE = "escalation_failure"
    ESCALATION_TOOL_LOOP = "escalation_tool_loop"
    ESCALATION_COMMAND_FAILURE = "escalation_command_failure"
    ESCALATION_BUILD_FAILURE = "escalation_build_failure"
    ESCALATION_SCHEMA_FAILURE = "escalation_schema_failure"
    ESCALATION_CONTEXT_OVERFLOW = "escalation_context_overflow"
    ESCALATION_HIGH_RISK = "escalation_high_risk"
    ESCALATION_VALIDATOR = "escalation_validator"
    FIXED_MODE = "fixed_mode"
    SHADOW_MODE = "shadow_mode"


class EventType(enum.StrEnum):
    CLASSIFICATION = "classification"
    DETERMINISTIC_OVERRIDE = "deterministic_override"
    CLASSIFIER_FALLBACK = "classifier_fallback"
    ROUTE_SELECTED = "route_selected"
    MODEL_PINNED = "model_pinned"
    PROVIDER_REQUEST = "provider_request"
    PROVIDER_FAILURE = "provider_failure"
    ESCALATION = "escalation"
    SENSITIVITY_WARNING = "sensitivity_warning"
    SENSITIVITY_BLOCK = "sensitivity_block"
    TASK_COMPLETED = "task_completed"
    TASK_CANCELLED = "task_cancelled"
    TASK_EXPIRED = "task_expired"


# ── Schemas ────────────────────────────────────────────────────────────


class ClassifierResult(BaseModel):
    """Strictly validated output from the classifier.

    Must never return a provider, model slug, endpoint,
    or credential. Every field is enum-constrained. Extra fields are rejected
    to prevent prompt injection via task text.
    """

    model_config = {"extra": "forbid"}

    task_class: TaskClass
    risk: RiskLevel
    sensitivity: Sensitivity
    requires_tools: bool = True
    requires_vision: bool = False
    long_context: bool = False
    destructive_potential: bool = False
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("confidence")
    @classmethod
    def confidence_range(cls, v: float) -> float:
        if v < 0.0 or v > 1.0:
            msg = f"Confidence must be between 0 and 1, got {v}"
            raise ValueError(msg)
        return v


class RouteSelection(BaseModel):
    """The result of policy evaluation: which destination to use and why."""

    primary_model: str
    escalation_model: str
    reason_code: ReasonCode
    task_class: TaskClass
    risk: RiskLevel
    sensitivity: Sensitivity
    confidence: float = 0.0
    classifier_raw: ClassifierResult | None = None


class ModelPin(BaseModel):
    """A pinned concrete model for a task session."""

    concrete_model: str
    provider: str = "openrouter"
    base_url: str = "https://openrouter.ai/api/v1"


class EscalationDecision(BaseModel):
    """Result of an escalation check."""

    should_escalate: bool
    reason_code: ReasonCode | None = None
    detail: str = ""


class TelemetryEvent(BaseModel):
    """Content-free structured event for observability."""

    event_type: EventType
    task_id: str | None = None
    session_id: str | None = None
    primary_model: str | None = None
    escalation_model: str | None = None
    concrete_model: str | None = None
    reason_code: ReasonCode | None = None
    latency_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    estimated_cost: float | None = None
    detail: str = ""


class HealthStatus(BaseModel):
    """Health check result for a component."""

    name: str
    healthy: bool
    detail: str = ""


class Destination(BaseModel):
    """A concrete model destination: provider + model slug.

    ``provider`` is explicit ("openrouter" or "ollama") — it is never
    inferred from the slug. OpenRouter destinations dispatch to the
    configured upstream; Ollama destinations dispatch to the native
    loopback Ollama API.
    """

    provider: str = "openrouter"
    model_slug: str
    retry_fallback: str | None = None
    """Optional destination slug to retry on when the primary fails
    transiently (429/5xx). Same provider as the primary by default."""


# ── Constants ──────────────────────────────────────────────────────────


# Task class -> (primary destination, escalation destination).
# The first element is used for normal/low-risk requests; the second is
# the escalation model used on high/critical risk. Routing is fully
# concrete — destinations are direct (provider, model) pairs, no aliases.
DEFAULT_ROUTE_TABLE: dict[TaskClass, tuple[Destination, Destination]] = {
    TaskClass.STRUCTURED_SIMPLE: (
        Destination(provider="openrouter", model_slug="openai/gpt-5.6-luna"),
        Destination(provider="openrouter", model_slug="z-ai/glm-5.2"),
    ),
    TaskClass.AGENTIC_EXECUTION: (
        Destination(provider="openrouter", model_slug="deepseek/deepseek-v4-flash-0731"),
        Destination(provider="openrouter", model_slug="openai/gpt-5.6-sol"),
    ),
    TaskClass.SOFTWARE_ENGINEERING: (
        Destination(provider="openrouter", model_slug="z-ai/glm-5.2"),
        Destination(provider="openrouter", model_slug="anthropic/claude-opus-5"),
    ),
    TaskClass.SECURITY_ENGINEERING: (
        Destination(provider="openrouter", model_slug="openai/gpt-5.6-sol"),
        Destination(
            provider="openrouter",
            model_slug="anthropic/claude-fable-5",
            retry_fallback="anthropic/claude-opus-5",
        ),
    ),
    TaskClass.KNOWLEDGE_REASONING: (
        Destination(provider="openrouter", model_slug="z-ai/glm-5.2"),
        Destination(provider="openrouter", model_slug="moonshotai/kimi-k3"),
    ),
    TaskClass.WRITING_COMMUNICATION: (
        Destination(provider="openrouter", model_slug="anthropic/claude-sonnet-5"),
        Destination(provider="openrouter", model_slug="anthropic/claude-opus-5"),
    ),
    TaskClass.COMPUTER_USE: (
        Destination(provider="openrouter", model_slug="anthropic/claude-sonnet-5"),
        Destination(provider="openrouter", model_slug="anthropic/claude-opus-5"),
    ),
    TaskClass.VISUAL_FRONTEND: (
        Destination(provider="openrouter", model_slug="moonshotai/kimi-k3"),
        Destination(provider="openrouter", model_slug="anthropic/claude-opus-5"),
    ),
}

# Destination used when no classification is possible (empty text,
# classifier unavailable, or a task class with no route).
DEFAULT_DESTINATION = Destination(
    provider="openrouter", model_slug="openai/gpt-5.6-luna"
)
