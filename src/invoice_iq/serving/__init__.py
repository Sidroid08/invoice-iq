"""FastAPI serving layer for invoice-iq."""

from invoice_iq.serving.app import create_app
from invoice_iq.serving.deps import AppDeps, build_app_deps

__all__ = ["AppDeps", "build_app_deps", "create_app"]
