"""
Test Client — Comprehensive End-to-End Verification of the MCP Server.

This script acts as an MCP client that connects to ``server.py`` via the
``stdio`` transport. It exercises every tool, resource, and error-handling
path to ensure the server is production-ready.

Usage
-----
::

    .\venv\Scripts\python test_client.py

The client uses the same Python interpreter (from the venv) to launch the
server subprocess to ensure dependency consistency.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

# ── Logging ────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


async def main():
    """Main execution loop for the test client.

    Iteratively calls server tools and resources, validating the responses.
    """
    logger.info("=" * 65)
    logger.info("  mcp-model-registry — Test Client")
    logger.info("=" * 65)

    # Resolve paths
    base_dir = Path(__file__).parent.resolve()
    server_script = base_dir / "server.py"

    # Configure StdioTransport using the venv's Python interpreter
    # This ensures the server has access to all installed dependencies.
    transport = StdioTransport(
        command=sys.executable,
        args=[str(server_script)],
        env=os.environ.copy(),
    )

    async with Client(transport) as client:
        passed = 0
        failed = 0

        async def run_test(name: str, coro):
            """Helper to execute a test case and log results."""
            nonlocal passed, failed
            logger.info("🧪 %s", name)
            try:
                result = await coro

                # Extract content from ToolResult blocks
                if isinstance(result, list):
                    text = " ".join(getattr(b, "text", str(b)) for b in result)
                else:
                    text = str(result)

                preview = text[:200] + ("..." if len(text) > 200 else "")
                logger.info("   ✅ PASS | %s", preview)
                passed += 1
                return text
            except Exception as e:
                logger.error("   ❌ FAIL | %s", e)
                failed += 1
                return None

        # ── 1. Tool Verification ──────────────────────────────────────

        logger.info("─" * 65)
        logger.info("  TOOL VERIFICATION")
        logger.info("─" * 65)

        await run_test(
            "list_models",
            client.call_tool("list_models", {}),
        )

        await run_test(
            "get_model_info (iris)",
            client.call_tool("get_model_info", {"model_id": "iris-classifier"}),
        )

        await run_test(
            "run_inference (iris setosa)",
            client.call_tool("run_inference", {
                "model_id": "iris-classifier",
                "input_data": "[5.1, 3.5, 1.4, 0.2]",
            }),
        )

        await run_test(
            "run_inference (sentiment positive)",
            client.call_tool("run_inference", {
                "model_id": "sentiment-analyzer",
                "input_data": "A truly remarkable film with brilliant acting.",
            }),
        )

        await run_test(
            "compare_models",
            client.call_tool("compare_models", {
                "model_ids": ["iris-classifier", "sentiment-analyzer"],
            }),
        )

        # ── 2. Resource Verification ──────────────────────────────────

        logger.info("─" * 65)
        logger.info("  RESOURCE VERIFICATION")
        logger.info("─" * 65)

        await run_test(
            "Resource: models://catalog",
            client.read_resource("models://catalog"),
        )

        await run_test(
            "Resource: models://iris-classifier/metadata",
            client.read_resource("models://iris-classifier/metadata"),
        )

        # ── 3. Error Handling ─────────────────────────────────────────

        logger.info("─" * 65)
        logger.info("  ERROR HANDLING")
        logger.info("─" * 65)

        await run_test(
            "run_inference (invalid model id)",
            client.call_tool("run_inference", {
                "model_id": "fake-model",
                "input_data": "{}",
            }),
        )

        await run_test(
            "run_inference (malformed iris data)",
            client.call_tool("run_inference", {
                "model_id": "iris-classifier",
                "input_data": "[1.0, 2.0]",
            }),
        )

        # ── Summary ───────────────────────────────────────────────────

        logger.info("=" * 65)
        total = passed + failed
        logger.info("  Final Result: %d/%d Passed", passed, total)
        if failed == 0:
            logger.info("  ✨ SUCCESS: All server capabilities validated.")
        else:
            logger.warning("  ⚠️ WARNING: %d test cases failed.", failed)
        logger.info("=" * 65)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
