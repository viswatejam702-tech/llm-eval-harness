"""
API Gateway entrypoint for Aegis Harness
Exposes the FastAPI application instance for Uvicorn / ASGI servers.
"""
import os
import sys

# Ensure repository root is on sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from serving.server import app

__all__ = ["app"]
