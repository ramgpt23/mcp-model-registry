"""
Model Registry — In-memory store for ML model metadata and lifecycle operations.

This module provides the ``ModelRegistry`` class, which acts as a lightweight,
in-memory registry for machine-learning models.  Each model's metadata is
captured in a ``ModelInfo`` dataclass, and the registry exposes methods to
register, list, inspect, load (from disk), and compare models.

The registry also provides JSON-serialisation helpers that the MCP server
uses to expose models as MCP resources.

Typical usage
-------------
>>> from models.registry import ModelRegistry, ModelInfo
>>> reg = ModelRegistry(models_dir="saved_models")
>>> reg.register(ModelInfo(
...     model_id="my-model", name="My Model", version="1.0.0",
...     framework="scikit-learn", task_type="classification",
...     description="A demo classifier.", file_path="my_model.joblib",
... ))
>>> reg.list_models()
[{'model_id': 'my-model', 'name': 'My Model', ...}]
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Metadata for a registered ML model.

    Attributes
    ----------
    model_id : str
        Unique identifier for the model (e.g. ``"iris-classifier"``).
    name : str
        Human-readable name shown to users and LLM clients.
    version : str
        Semantic version string (e.g. ``"1.0.0"``).
    framework : str
        ML framework used (e.g. ``"scikit-learn"``, ``"pytorch"``).
    task_type : str
        Type of ML task (e.g. ``"classification"``, ``"nlp-classification"``).
    description : str
        Free-text description of what the model does.
    file_path : str
        Filename of the serialised model relative to the ``models_dir``.
    metrics : dict[str, float]
        Evaluation metrics recorded after training (accuracy, F1, etc.).
    input_schema : dict[str, Any]
        JSON-Schema-style description of the expected input format.
    tags : list[str]
        Freeform tags for filtering and organisation.
    created_at : str
        ISO-8601 timestamp of when the model was registered.
    """

    model_id: str
    name: str
    version: str
    framework: str
    task_type: str
    description: str
    file_path: str
    metrics: dict[str, float] = field(default_factory=dict)
    input_schema: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialise the model info to a plain dictionary."""
        return asdict(self)

    def summary(self) -> dict[str, Any]:
        """Return a compact summary suitable for listing endpoints.

        Returns
        -------
        dict[str, Any]
            Dictionary with ``model_id``, ``name``, ``version``,
            ``task_type``, ``framework``, and ``tags``.
        """
        return {
            "model_id": self.model_id,
            "name": self.name,
            "version": self.version,
            "task_type": self.task_type,
            "framework": self.framework,
            "tags": self.tags,
        }


class ModelRegistry:
    """In-memory model registry with file-backed model loading.

    Parameters
    ----------
    models_dir : str
        Directory where serialised model files (e.g. ``.joblib``) reside.
        The directory is created automatically if it does not exist.

    Examples
    --------
    >>> registry = ModelRegistry("saved_models")
    >>> registry.register(ModelInfo(...))
    >>> registry.list_models()
    """

    def __init__(self, models_dir: str = "saved_models"):
        self._models: dict[str, ModelInfo] = {}
        self._models_dir = Path(models_dir)
        self._models_dir.mkdir(parents=True, exist_ok=True)
        logger.info("ModelRegistry initialised — models_dir=%s", self._models_dir)

    def register(self, info: ModelInfo) -> ModelInfo:
        """Register a model in the registry.

        Parameters
        ----------
        info : ModelInfo
            Metadata for the model to register.

        Returns
        -------
        ModelInfo
            The same ``ModelInfo`` instance, for chaining.
        """
        self._models[info.model_id] = info
        logger.info("Registered model '%s' (v%s)", info.model_id, info.version)
        return info

    def list_models(self) -> list[dict[str, Any]]:
        """Return summary list of all registered models.

        Returns
        -------
        list[dict[str, Any]]
            Each item is the compact summary from ``ModelInfo.summary()``.
        """
        return [m.summary() for m in self._models.values()]

    def get_model(self, model_id: str) -> ModelInfo | None:
        """Look up full metadata for a model by its ID.

        Parameters
        ----------
        model_id : str
            Unique identifier of the model.

        Returns
        -------
        ModelInfo or None
            The model's metadata, or ``None`` if not found.
        """
        return self._models.get(model_id)

    def get_all_ids(self) -> list[str]:
        """Return all registered model IDs.

        Returns
        -------
        list[str]
        """
        return list(self._models.keys())

    def load_model(self, model_id: str) -> Any:
        """Deserialise and return a trained model from disk.

        Parameters
        ----------
        model_id : str
            Unique identifier of the model.

        Returns
        -------
        Any
            The deserialised model object (e.g. a scikit-learn estimator).

        Raises
        ------
        ValueError
            If the ``model_id`` is not found in the registry.
        FileNotFoundError
            If the serialised file does not exist on disk.
        """
        info = self.get_model(model_id)
        if info is None:
            raise ValueError(f"Model '{model_id}' not found in registry.")

        path = self._models_dir / info.file_path
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        logger.info("Loading model '%s' from %s", model_id, path)
        return joblib.load(path)

    def compare_models(self, model_ids: list[str]) -> dict[str, Any]:
        """Compare metrics across multiple models.

        Parameters
        ----------
        model_ids : list[str]
            IDs of the models to compare.

        Returns
        -------
        dict[str, Any]
            Mapping of ``model_id`` → ``{name, version, task_type, metrics, framework}``
            (or an ``error`` key if the model is not found).
        """
        comparison = {}
        for mid in model_ids:
            info = self.get_model(mid)
            if info is None:
                comparison[mid] = {"error": f"Model '{mid}' not found"}
            else:
                comparison[mid] = {
                    "name": info.name,
                    "version": info.version,
                    "task_type": info.task_type,
                    "metrics": info.metrics,
                    "framework": info.framework,
                }
        return comparison

    def catalog_json(self) -> str:
        """Return JSON catalog of all models (for MCP resource).

        Returns
        -------
        str
            Pretty-printed JSON string.
        """
        catalog = {
            "total_models": len(self._models),
            "models": [m.to_dict() for m in self._models.values()],
        }
        return json.dumps(catalog, indent=2)

    def model_metadata_json(self, model_id: str) -> str:
        """Return JSON metadata for a specific model (for MCP resource).

        Parameters
        ----------
        model_id : str
            Unique identifier of the model.

        Returns
        -------
        str
            Pretty-printed JSON string of the model's metadata,
            or an error payload if not found.
        """
        info = self.get_model(model_id)
        if info is None:
            return json.dumps({"error": f"Model '{model_id}' not found"})
        return json.dumps(info.to_dict(), indent=2)
