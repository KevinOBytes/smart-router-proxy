"""BERT prompt classifier — fast local intent detection.

Sits between deterministic rules and Gemma LLM classification.
Loads the trained distilbert + MLX head model once and caches it.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from transformers import AutoTokenizer

from smart_router_proxy.models import (
    ClassifierResult,
    RiskLevel,
    Sensitivity,
    TaskClass,
)

logger = logging.getLogger(__name__)

# ── Model path ──────────────────────────────────────────────────────────
MODEL_DIR = os.path.expanduser("~/.smart-router-proxy/classifier-model")

# ── BERT categories → proxy TaskClass mapping ──────────────────────────
BERT_TO_TASKCLASS: dict[str, TaskClass] = {
    "coding": TaskClass.SOFTWARE_ENGINEERING,
    "cli": TaskClass.AGENTIC_EXECUTION,
    "computer_use": TaskClass.COMPUTER_USE,
    "general_knowledge": TaskClass.KNOWLEDGE_REASONING,
    "roleplay": TaskClass.WRITING_COMMUNICATION,
    "math": TaskClass.STRUCTURED_SIMPLE,
    "writing_editing": TaskClass.WRITING_COMMUNICATION,
    "data_analysis": TaskClass.STRUCTURED_SIMPLE,
    "security_threat": TaskClass.SECURITY_ENGINEERING,
    "planning": TaskClass.AGENTIC_EXECUTION,
}

# ── Risk/sensitivity heuristics per category ───────────────────────────
CATEGORY_RISK: dict[str, RiskLevel] = {
    "coding": RiskLevel.MODERATE,
    "cli": RiskLevel.MODERATE,
    "computer_use": RiskLevel.MODERATE,
    "general_knowledge": RiskLevel.LOW,
    "roleplay": RiskLevel.LOW,
    "math": RiskLevel.LOW,
    "writing_editing": RiskLevel.LOW,
    "data_analysis": RiskLevel.LOW,
    "security_threat": RiskLevel.HIGH,
    "planning": RiskLevel.LOW,
}

CATEGORY_SENSITIVITY: dict[str, Sensitivity] = {
    "coding": Sensitivity.INTERNAL,
    "cli": Sensitivity.INTERNAL,
    "computer_use": Sensitivity.INTERNAL,
    "general_knowledge": Sensitivity.PUBLIC,
    "roleplay": Sensitivity.PUBLIC,
    "math": Sensitivity.PUBLIC,
    "writing_editing": Sensitivity.INTERNAL,
    "data_analysis": Sensitivity.INTERNAL,
    "security_threat": Sensitivity.CONFIDENTIAL,
    "planning": Sensitivity.INTERNAL,
}

CATEGORY_DESTRUCTIVE: set[str] = {"security_threat"}

# ── Confidence threshold ───────────────────────────────────────────────
# Below this, defer to Gemma
CONFIDENCE_THRESHOLD = 0.45


class ClassifierHead(nn.Module):
    """Tiny MLP on top of frozen distilbert features."""

    def __init__(self, input_dim: int = 768, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.net(x)


# ── Singleton ──────────────────────────────────────────────────────────
_classifier: BertClassifier | None = None


def get_classifier() -> BertClassifier:
    global _classifier
    if _classifier is None:
        _classifier = BertClassifier()
    return _classifier


class BertClassifier:
    """Wraps the trained distilbert + MLX classifier head."""

    def __init__(self, model_dir: str | Path = MODEL_DIR):
        model_dir = str(model_dir)
        config_path = os.path.join(model_dir, "config.json")
        weights_path = os.path.join(model_dir, "weights.safetensors")

        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"BERT model not found at {model_dir}. "
                "Train first with ~/Projects/model-train/train_mlx.py"
            )

        with open(config_path) as f:
            self.config = json.load(f)

        self.id2label = {int(k): v for k, v in self.config["id2label"].items()}
        self.num_classes = self.config["num_classes"]
        self.max_length = self.config.get("max_length", 64)

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # MLX classifier head
        self.model = ClassifierHead(num_classes=self.num_classes)
        self.model.load_weights(weights_path)
        mx.eval(self.model.parameters())

        # Cached HF feature extractor
        from transformers import AutoModel

        self.hf_model = AutoModel.from_pretrained(self.config["base_model"])
        self.hf_model.eval()

    def _extract_features(self, texts: list[str]) -> mx.array:
        """Tokenize and get [CLS] embeddings via distilbert."""
        import torch

        all_features = []
        for text in texts:
            enc = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            with torch.no_grad():
                outputs = self.hf_model(**enc)
                cls_embeds = outputs.last_hidden_state[:, 0, :].numpy()
            all_features.append(cls_embeds)

        features = np.concatenate(all_features, axis=0)
        return mx.array(features)

    def classify(self, text: str) -> tuple[str, float]:
        """Return (category_label, confidence)."""
        features = self._extract_features([text])
        logits = self.model(features)
        probs = mx.softmax(logits, axis=1)
        pred_idx = mx.argmax(logits, axis=1).item()
        confidence = probs[0, pred_idx].item()
        return self.id2label[pred_idx], confidence

    def classify_to_result(self, text: str) -> ClassifierResult | None:
        """Classify and return a ClassifierResult, or None if below threshold."""
        label, confidence = self.classify(text)

        if confidence < CONFIDENCE_THRESHOLD:
            logger.debug("BERT confidence %.2f below threshold, deferring", confidence)
            return None

        task_class = BERT_TO_TASKCLASS.get(label)
        if task_class is None:
            logger.debug("BERT label '%s' has no TaskClass mapping, deferring", label)
            return None

        destructive = label in CATEGORY_DESTRUCTIVE

        return ClassifierResult(
            task_class=task_class,
            risk=CATEGORY_RISK.get(label, RiskLevel.LOW),
            sensitivity=CATEGORY_SENSITIVITY.get(label, Sensitivity.INTERNAL),
            requires_tools=True,
            requires_vision=task_class == TaskClass.COMPUTER_USE,
            long_context=False,
            destructive_potential=destructive,
            confidence=confidence,
        )
