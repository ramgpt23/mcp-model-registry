"""
mcp-model-registry — Lifecycle Management of ML Models via MCP.

This MCP server provides a standardized interface for managing, inspecting,
and using machine learning models. Built with the **FastMCP** SDK, it exposes
models as MCP tools, resources, and prompts.

Features
--------
- **Tools**: List models, get detailed info, run inference, compare models,
  and fetch metrics.
- **Resources**: Browse the complete model catalog or read specific model
  metadata via URI templates.
- **Prompts**: Reusable patterns for analyzing predictions and comparing models.

Architecture
------------
The server initializes an internal ``ModelRegistry`` which loads serialized
models (``joblib``) from the ``saved_models/`` directory. Clients connect
via stdio or HTTP transport.

Usage
-----
To run as a stdio server (default for most MCP clients like Claude):
::

    python server.py

To run as an HTTP server:
::

    python server.py --transport http --port 8080
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any
from fastmcp import FastMCP
from fastapi import FastAPI
from starlette.responses import JSONResponse
from models.registry import ModelRegistry, ModelInfo

# ── Logging ────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,  # Crucial for stdio transport to avoid polluting stdout
)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# Initialize Registry & Register Demo Models
# ══════════════════════════════════════════════════════════════════════════════

registry = ModelRegistry(models_dir="saved_models")


def _bootstrap_registry():
    """Register production-ready demo models in the registry."""

    # ── Iris Classifier ──
    registry.register(ModelInfo(
        model_id="iris-classifier",
        name="Iris Flower Classifier",
        version="1.1.0",
        framework="scikit-learn",
        task_type="classification",
        description=(
            "A RandomForestClassifier trained on the real Fisher's Iris dataset. "
            "Classifies flowers into setosa, versicolor, or virginica based "
            "on 4 numerical measurements (sepal/petal length/width)."
        ),
        file_path="iris_classifier.joblib",
        metrics={
            "accuracy": 1.0,
            "f1_weighted": 1.0,
            "precision_weighted": 1.0,
            "recall_weighted": 1.0,
        },
        input_schema={
            "type": "array",
            "description": "4 numeric features: [sepal_length, sepal_width, petal_length, petal_width]",
            "items": {"type": "number"},
            "minItems": 4,
            "maxItems": 4,
            "example": [5.1, 3.5, 1.4, 0.2],
        },
        tags=["classification", "sklearn", "iris", "botany"],
    ))

    # ── Sentiment Analyzer ──
    registry.register(ModelInfo(
        model_id="sentiment-analyzer",
        name="IMDB Movie Review Sentiment Analyzer",
        version="1.1.0",
        framework="scikit-learn",
        task_type="nlp-classification",
        description=(
            "A TF-IDF + LogisticRegression pipeline trained on 15,000 real IMDB reviews. "
            "Classifies movie reviews as positive (1) or negative (0). Highly robust."
        ),
        file_path="sentiment_analyzer.joblib",
        metrics={
            "accuracy": 0.8752,
            "f1_binary": 0.8726,
            "precision": 0.8841,
            "recall": 0.8614,
        },
        input_schema={
            "type": "string",
            "description": "Review text to classify (e.g. 'This film was spectacular!')",
            "example": "This movie was absolutely fantastic!",
        },
        tags=["nlp", "sentiment", "sklearn", "imdb", "production-quality"],
    ))


_bootstrap_registry()


# ══════════════════════════════════════════════════════════════════════════════
# Core Logic (Shared Helpers)
# ══════════════════════════════════════════════════════════════════════════════

def _run_inference_logic(model_id: str, input_data: str) -> Any:
    """Shared core logic for model inference."""
    import numpy as np

    info = registry.get_model(model_id)
    if info is None:
        return {"error": f"Model '{model_id}' not found."}

    try:
        model = registry.load_model(model_id)
    except Exception as e:
        logger.error("Failed to load model %s: %s", model_id, e)
        return {"error": str(e), "hint": "Ensure models are trained via train_demo_models.py"}

    try:
        # ── Iris Classifier ──
        if model_id == "iris-classifier":
            features = json.loads(input_data)
            if not isinstance(features, list) or len(features) != 4:
                return {"error": "Iris model expects exactly 4 numeric features."}

            features_array = np.array(features).reshape(1, -1)
            prediction = model.predict(features_array)[0]
            probabilities = model.predict_proba(features_array)[0]
            class_names = ["setosa", "versicolor", "virginica"]

            return {
                "model_id": model_id,
                "prediction": class_names[prediction],
                "confidence": round(float(max(probabilities)), 4),
                "probabilities": {name: round(float(p), 4) for name, p in zip(class_names, probabilities)}
            }

        # ── Sentiment Analyzer ──
        elif model_id == "sentiment-analyzer":
            text = input_data.strip().strip('"').strip("'")
            prediction = model.predict([text])[0]
            probabilities = model.predict_proba([text])[0]
            labels = ["negative", "positive"]

            return {
                "model_id": model_id,
                "sentiment": labels[prediction],
                "confidence": round(float(max(probabilities)), 4),
                "probabilities": {name: round(float(p), 4) for name, p in zip(labels, probabilities)}
            }
        else:
            return {"error": f"Inference logic for '{model_id}' is not implemented."}

    except Exception as e:
        logger.exception("Inference failed for %s", model_id)
        return {"error": f"Inference engine error: {str(e)}"}


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI Application Setup (Routes Defined First for Swagger)
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="mcp-model-registry",
    description="MCP server for ML model lifecycle management — list, inspect, compare, and run inference on models via natural language.",
    version="1.0.0"
)


@app.get("/", tags=["Internal"], summary="Service Welcome & Capabilities")
def root_endpoint():
    """
    Returns a welcome message and lists the server's current capabilities
    (Tools, Resources, Prompts). Use this to verify the server is responding.
    """
    return JSONResponse({
        "status": "online",
        "service": "mcp-model-registry",
        "description": "Lifecycle Management of ML Models via MCP",
        "capabilities": ["tools", "resources", "prompts"],
        "docs": "/docs"
    })


@app.get("/health", tags=["Internal"], summary="Service Health Check")
def health_check():
    """
    Performs a deep health check of the service, including checking the
    status of the underlying Model Registry and the number of loaded models.
    """
    return JSONResponse({
        "status": "healthy",
        "registry_models": len(registry.list_models())
    })


@app.get("/tools/list_models", tags=["Registry"], summary="List All Registered Models")
def list_models_api() -> Any:
    """
    Returns an array of all machine learning models currently registered in the system.
    Each entry includes summary fields like name, version, and task type.
    """
    return registry.list_models()


@app.get("/tools/get_model_info/{model_id}", tags=["Registry"], summary="Get Detailed Model Metadata")
def get_model_info_api(model_id: str) -> Any:
    """
    Retrieves full architectural metadata for a specific model ID.
    Includes technical specifics like framework version, training parameters,
    and expected input schemas.
    """
    info = registry.get_model(model_id)
    if info is None:
        return {"error": f"Model '{model_id}' not found."}
    return info.to_dict()


@app.post("/tools/run_inference/{model_id}", tags=["Registry"], summary="Run Real-Time Inference")
def run_inference_api(model_id: str, input_data: str) -> Any:
    """
    Executes a model prediction on the provided user data.

    **Input Formatting:**
    * **iris-classifier**: Expects a JSON array of 4 floats, e.g., `[5.1, 3.5, 1.4, 0.2]`
    * **sentiment-analyzer**: Expects a plain text string review.

    Returns the prediction result along with confidence scores/probabilities.
    """
    return _run_inference_logic(model_id, input_data)


@app.post("/tools/compare_models", tags=["Registry"], summary="Cross-Model Comparison")
def compare_models_api(model_ids: list[str]) -> Any:
    """
    Takes a list of multiple model IDs and generates a side-by-side performance
    comparison. Useful for selecting the best model version for a specific task.
    """
    if len(model_ids) < 2:
        return {"error": "Minimum 2 models required for comparison."}
    return registry.compare_models(model_ids)


@app.get("/tools/get_model_metrics/{model_id}", tags=["Registry"], summary="Fetch Model Performance Metrics")
def get_model_metrics_api(model_id: str) -> Any:
    """
    Retrieves the 'report card' for a specific model, containing accuracy,
    precision, recall, and F1 scores recorded during validation.
    """
    info = registry.get_model(model_id)
    if info is None:
        return {"error": f"Model '{model_id}' not found."}
    return {"model_id": model_id, "metrics": info.metrics}


@app.get("/resources/catalog", tags=["Registry"], summary="Download Model Catalog (JSON)")
def get_model_catalog_api() -> Any:
    """
    Returns the complete system catalog in a single JSON structure.
    Used for bulk processing or syncing with external monitoring tools.
    """
    return registry.list_models()


@app.get("/resources/metadata/{model_id}", tags=["Registry"], summary="Direct Metadata Resource Access")
def get_model_metadata_api(model_id: str) -> Any:
    """
    Provides direct resource-level access to a model's internal metadata object.
    Mirrors the behavior of the MCP resource 'models://{id}/metadata'.
    """
    info = registry.get_model(model_id)
    if info is None:
        return {"error": f"Model '{model_id}' not found."}
    return info.to_dict()


# ══════════════════════════════════════════════════════════════════════════════
# MCP Setup (Mounted onto FastAPI)
# ══════════════════════════════════════════════════════════════════════════════

mcp = FastMCP.from_fastapi(
    app,
    name="mcp-model-registry",
    instructions=(
        "You are an expert ML Operations assistant. Use the available tools to "
        "help users explore, compare, and run inference on machine learning models. "
        "When running inference, always provide a detailed explanation of the "
        "result using the `analyze_prediction` prompt pattern."
    ),
)


@mcp.tool()
def list_models() -> str:
    """List all registered ML models with their summary info."""
    models = registry.list_models()
    return json.dumps(models, indent=2)


@mcp.tool()
def get_model_info(model_id: str) -> str:
    """Get detailed metadata for a specific model."""
    info = registry.get_model(model_id)
    if info is None:
        return json.dumps({"error": f"Model '{model_id}' not found."})
    return json.dumps(info.to_dict(), indent=2)


@mcp.tool()
def run_inference(model_id: str, input_data: str) -> str:
    """Run a real-time prediction using a registered model."""
    result = _run_inference_logic(model_id, input_data)
    if isinstance(result, dict) and "error" in result:
        return json.dumps(result)
    return json.dumps(result, indent=2)


@mcp.tool()
def compare_models(model_ids: list[str]) -> str:
    """Compare performance metrics across multiple models."""
    if len(model_ids) < 2:
        return json.dumps({"error": "Minimum 2 models required for comparison."})
    comparison = registry.compare_models(model_ids)
    return json.dumps(comparison, indent=2)


@mcp.tool()
def get_model_metrics(model_id: str) -> str:
    """Retrieve detailed evaluation metrics for a specific model."""
    info = registry.get_model(model_id)
    if info is None:
        return json.dumps({"error": f"Model '{model_id}' not found."})
    return json.dumps({"model_id": model_id, "metrics": info.metrics}, indent=2)


@mcp.resource("models://catalog")
def get_model_catalog() -> str:
    """Full architectural catalog of all registered models."""
    return registry.catalog_json()


@mcp.resource("models://{model_id}/metadata")
def get_model_metadata(model_id: str) -> str:
    """Direct resource link to a specific model's metadata."""
    return registry.model_metadata_json(model_id)


@mcp.prompt()
def analyze_prediction(model_name: str, prediction_result: str) -> str:
    """Templates for explaining ML predictions to non-technical users."""
    return (
        f"Explain this result from the '{model_name}' model in clear English:\n"
        f"```json\n{prediction_result}\n```\n"
        "Focus on confidence, what the labels mean, and next steps."
    )


# ══════════════════════════════════════════════════════════════════════════════
# Execution Entry Point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    import os

    # Check for HTTP transport via env or CLI flags
    is_http = os.environ.get("MCP_TRANSPORT") == "http" or "--transport" in sys.argv

    if is_http:
        logger.info("🚀 Starting mcp-model-registry via FastAPI/Uvicorn (HTTP)")
        port = 8080
        if "--port" in sys.argv:
            try:
                port = int(sys.argv[sys.argv.index("--port") + 1])
            except (ValueError, IndexError):
                pass
        uvicorn.run(app, host="0.0.0.0", port=port)
    else:
        # Default to Stdio for MCP clients
        logger.info("🚀 Starting mcp-model-registry MCP Server (Stdio)")
        mcp.run()
