# 🗄️ mcp-model-registry

[![MCP](https://img.shields.io/badge/MCP-Protocol-blue)](https://modelcontextprotocol.io/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange)](https://scikit-learn.org/)

 **Model Context Protocol (MCP)** server that exposes Machine Learning model lifecycle operations—listing, inspecting, comparing, and running inference—directly to LLM clients (like Claude Desktop, Cursor, and IDEs).

This project demonstrates how to bridge the gap between Large Language Models and specialized, local ML models through a standardized protocol.

---

## ✨ Features

- **Standardized Model Operations**: Expose any scikit-learn (or other) model via a unified tool interface.
- **Real-Time Inference**: Run predictions on real datasets through natural language queries.
- **Model Registry**: A robust in-memory metadata store tracking versions, task types, and frameworks.
- **Deep Comparison**: Side-by-side metric evaluation (Accuracy, F1, Precision, Recall).
- **Rich Context**: Exposes models as MCP Resources and provides Prompts for prediction analysis.
- **Portfolio-Ready**: Includes real IMDB and Iris datasets, professional logging, and full type hinting.

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- `uv` (recommended) or `pip`

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/ramgpt23/mcp-model-registry.git
cd mcp-model-registry
```

You can install the dependencies using either `uv` (recommended for speed) or standard `pip`.

#### Option A: Using `uv` (Recommended)
```bash
# Install dependencies into a managed environment
uv sync
```

#### Option B: Using `pip`
```bash
# Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install from requirements.txt
pip install -r requirements.txt

# Alternatively, install the project in editable mode
pip install -e .
```

### 3. Train the Demo Models
This script downloads the **real IMDB dataset** and trains the demo models.
```bash
python models/train_demo_models.py
```

### 4. Run the Server
```bash
# Start via stdio (Default for MCP)
python server.py

# Run via HTTP (with FastAPI Swagger UI)
python server.py --transport http --port 8080

# The HTTP server now exposes:
# - Root: http://localhost:8080/
# - Health: http://localhost:8080/health
# - Swagger Docs: http://localhost:8080/docs
```

---

## 🌐 HTTP API Reference

When running in HTTP mode, the server exposes a full FastAPI-powered web interface.

### General Endpoints
| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Root welcome message and service status. |
| `/health` | `GET` | Service health check and model count. |
| `/docs` | `GET` | **Swagger UI** (interactive API documentation). |
| `/openapi.json` | `GET` | Raw OpenAPI specification. |

### Model Registry (Tools)
| Endpoint | Method | Description |
|---|---|---|
| `/tools/list_models` | `GET` | List all available models. |
| `/tools/get_model_info/{id}`| `GET` | Get detailed metadata for a specific model. |
| `/tools/run_inference/{id}` | `POST` | Run a real-time prediction. |
| `/tools/compare_models` | `POST` | Compare multiple models side-by-side. |
| `/tools/get_model_metrics/{id}`| `GET` | Fetch evaluation scores (F1, Accuracy, etc.). |

### MCP Resources
| Endpoint | Method | Description |
|---|---|---|
| `/resources/catalog` | `GET` | Complete JSON inventory of the model registry. |
| `/resources/metadata/{id}` | `GET` | Direct JSON metadata for a specific model. |

---

## 🛠️ MCP Interface

### Tools
| Tool | Description |
|---|---|
| `list_models` | List all available ML models. |
| `get_model_info` | Get detailed metadata/metrics for a specific model. |
| `run_inference` | Run a real-time prediction on a model. |
| `compare_models` | Compare performance across multiple models. |
| `get_model_metrics` | Fetch evaluation scores (F1, Accuracy, etc.). |

### Resources
- `models://catalog`: A complete JSON inventory of the model registry.
- `models://{model_id}/metadata`: Direct access to a specific model's metadata.

---

## 🧪 Demo Models Included

1. **Iris Flower Classifier**:
   - **Dataset**: Fisher's Iris (150 samples).
   - **Task**: 3-class classification (Setosa, Versicolor, Virginica).
   - **Framework**: Scikit-learn (RandomForest).

2. **IMDB Sentiment Analyzer**:
   - **Dataset**: IMDB Movie Reviews (50,000 real reviews).
   - **Task**: Binary sentiment classification (Positive/Negative).
   - **Framework**: TF-IDF + Logistic Regression.

---

## 📈 Verification
Run the included test client to verify the full MCP lifecycle:
```bash
python test_client.py
```

---

## 📝 License
MIT
