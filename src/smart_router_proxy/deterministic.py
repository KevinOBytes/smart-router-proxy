"""High-confidence deterministic classification rules.

Runs before Gemma to handle unambiguous cases. Requires strong structural
evidence and defers to Gemma when uncertain.
"""

from __future__ import annotations

import re

from smart_router_proxy.models import (
    ClassifierResult,
    RiskLevel,
    Sensitivity,
    TaskClass,
)

# ── Structural patterns ────────────────────────────────────────────────
# Patterns use \s+(?:\S+\s+)*? to allow intervening words between verb and noun

_SHELL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(?:run|execute|deploy|install|setup|configure)\s+(?:\S+\s+)*?(?:command|script|shell|bash|zsh|sh)\b", re.I),
    re.compile(r"\b(?:apt|brew|pip|npm|yarn|docker|kubectl|helm|terraform|ansible)\s+(?:install|update|upgrade|remove|deploy|apply)\b", re.I),
    re.compile(r"\b(?:deploy|rollout|release)\s+(?:\S+\s+)*?(?:to|on)\s+(?:\S+\s+)*?(?:production|staging|dev)\b", re.I),
    re.compile(r"\b(?:rm\s+-rf|dd\s+if=|format\s+(?:drive|disk|usb)|mkfs|fdisk|shutdown|reboot)\b", re.I),
]

_REPO_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(?:refactor|implement|write|create|modify|update|fix|add|remove|delete)\s+(?:\S+\s+)*?(?:function|class|method|module|file|test|route|endpoint|api|component|widget|code|logic|pipeline|workflow|service|handler|controller|middleware|schema|migration)\b", re.I),
    re.compile(r"\b(?:commit|push|pr|pull.request|merge|branch|checkout)\b", re.I),
    re.compile(r"\b(?:build|compile|test.suite|unit.test|integration.test|e2e)\b", re.I),
]

_SECURITY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(?:vulnerability|exploit|cve-\d+|cwe-\d+|malware|ransomware|backdoor|trojan|rootkit)\b", re.I),
    re.compile(r"\b(?:detection|detect|hunt|hunting|threat\s+intel|indicator\s+of\s+compromise|ioc|yara|sigma|kql|spl)\b", re.I),
    re.compile(r"\b(?:dfir|forensic|incident\s+response|remediate|containment)\b", re.I),
    re.compile(r"\b(?:cryptography|cipher|tls\s+certificate|ssl\s+certificate|oauth\s+flow|saml\s+assertion|jwt\s+token)\b", re.I),
]

_WRITING_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(?:write|draft|compose|author|create)\s+(?:\S+\s+)*?(?:report|memo|email|article|blog|post|documentation|doc|readme|guide|tutorial|page|newsletter|brief|summary)\b", re.I),
    re.compile(r"\b(?:summarize|summarise|outline|rewrite|proofread|edit|review)\s+(?:\S+\s+)*?(?:document|text|content|article|page|section|chapter|paper)\b", re.I),
]

_VISION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(?:ui|ux|frontend|front.end|interface|layout|design|mockup|wireframe|prototype)\b", re.I),
    re.compile(r"\b(?:diagram|chart|graph|plot|visualization|illustration|image|picture|screenshot|render)\b", re.I),
    re.compile(r"\b(?:css|html|svg|canvas|webgl|three\.js|d3\.js|chart\.js)\b", re.I),
]

_COMPUTER_USE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(?:click|type|scroll|navigate|browse|open\s+(?:app|application|browser|finder|terminal))\b", re.I),
    re.compile(r"\b(?:gui|desktop|window|screen|mouse|cursor|keyboard)\b", re.I),
    re.compile(r"\b(?:computer.use|computer_use|cua.driver)\b", re.I),
]

_VOICE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(?:tts|text.to.speech|speech|voice|audio\s+output|say|speak|read\s+aloud)\b", re.I),
    re.compile(r"\b(?:stt|speech.to.text|transcribe|transcription|audio\s+input|listen|hear)\b", re.I),
]

_STRUCTURED_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(?:extract|parse|convert|transform|validate)\s+(?:\S+\s+)*?(?:json|xml|csv|yaml|toml|config|data|schema|format|file)\b", re.I),
    re.compile(r"\b(?:classify|categorize|tag|label)\s+(?:\S+\s+)*?(?:text|content|item|record|row|entry|data|file)\b", re.I),
    re.compile(r"\b(?:summarize|summarise)\s+(?:\S+\s+)*?(?:json|data|log|file|content)\s+(?:as|into)\s+(?:json|yaml|table)\b", re.I),
]


def classify_deterministic(
    user_request: str,
    tool_names: list[str] | None = None,
) -> ClassifierResult | None:
    """Attempt deterministic classification.

    Returns a ClassifierResult only when structural evidence is strong
    and unambiguous. Returns None to defer to Gemma.

    Args:
        user_request: The initial user request text.
        tool_names: Names of tools available in the session.

    Returns:
        ClassifierResult if deterministically classified, None to defer.
    """
    text = user_request.strip()
    if not text:
        return None

    # Computer use: if computer_use tools are present, force it
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

    # Check structural patterns in priority order
    # Security takes priority over general software engineering
    if _strong_match(text, _SECURITY_PATTERNS):
        destructive = bool(re.search(r"\b(?:exploit|malware|backdoor|rootkit)\b", text, re.I))
        return ClassifierResult(
            task_class=TaskClass.SECURITY_ENGINEERING,
            risk=RiskLevel.HIGH if destructive else RiskLevel.MODERATE,
            sensitivity=Sensitivity.CONFIDENTIAL if destructive else Sensitivity.INTERNAL,
            requires_tools=True,
            requires_vision=False,
            long_context=False,
            destructive_potential=destructive,
            confidence=0.90,
        )

    if _strong_match(text, _SHELL_PATTERNS):
        destructive = bool(re.search(r"\b(?:rm\s+-rf|dd\s+if=|format|mkfs|fdisk)\b", text, re.I))
        return ClassifierResult(
            task_class=TaskClass.AGENTIC_EXECUTION,
            risk=RiskLevel.HIGH if destructive else RiskLevel.MODERATE,
            sensitivity=Sensitivity.INTERNAL,
            requires_tools=True,
            requires_vision=False,
            long_context=False,
            destructive_potential=destructive,
            confidence=0.90,
        )

    if _strong_match(text, _REPO_PATTERNS):
        return ClassifierResult(
            task_class=TaskClass.SOFTWARE_ENGINEERING,
            risk=RiskLevel.MODERATE,
            sensitivity=Sensitivity.INTERNAL,
            requires_tools=True,
            requires_vision=False,
            long_context=False,
            destructive_potential=False,
            confidence=0.85,
        )

    if _strong_match(text, _WRITING_PATTERNS):
        return ClassifierResult(
            task_class=TaskClass.WRITING_COMMUNICATION,
            risk=RiskLevel.LOW,
            sensitivity=Sensitivity.INTERNAL,
            requires_tools=False,
            requires_vision=False,
            long_context=False,
            destructive_potential=False,
            confidence=0.85,
        )

    if _strong_match(text, _VISION_PATTERNS):
        return ClassifierResult(
            task_class=TaskClass.VISUAL_FRONTEND,
            risk=RiskLevel.LOW,
            sensitivity=Sensitivity.PUBLIC,
            requires_tools=True,
            requires_vision=True,
            long_context=False,
            destructive_potential=False,
            confidence=0.80,
        )

    if _strong_match(text, _STRUCTURED_PATTERNS):
        return ClassifierResult(
            task_class=TaskClass.STRUCTURED_SIMPLE,
            risk=RiskLevel.LOW,
            sensitivity=Sensitivity.PUBLIC,
            requires_tools=False,
            requires_vision=False,
            long_context=False,
            destructive_potential=False,
            confidence=0.85,
        )

    # Computer use from text patterns
    if _strong_match(text, _COMPUTER_USE_PATTERNS):
        return ClassifierResult(
            task_class=TaskClass.COMPUTER_USE,
            risk=RiskLevel.MODERATE,
            sensitivity=Sensitivity.INTERNAL,
            requires_tools=True,
            requires_vision=True,
            long_context=False,
            destructive_potential=False,
            confidence=0.85,
        )

    # Voice/TTS/STT tasks route to Luna (supports audio)
    if _strong_match(text, _VOICE_PATTERNS):
        return ClassifierResult(
            task_class=TaskClass.STRUCTURED_SIMPLE,
            risk=RiskLevel.LOW,
            sensitivity=Sensitivity.PUBLIC,
            requires_tools=False,
            requires_vision=False,
            long_context=False,
            destructive_potential=False,
            confidence=0.80,
        )

    return None


def _strong_match(text: str, patterns: list[re.Pattern[str]]) -> bool:
    """Check if text has a structural match against patterns.

    Each pattern is already specific enough that a single match is
    strong evidence. Returns True if at least one pattern matches.
    """
    return any(p.search(text) for p in patterns)
