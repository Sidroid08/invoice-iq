"""Run the local invoice-iq demo.

Usage:
    python scripts/demo.py
    python scripts/demo.py --pdf data/samples/sample_invoice.pdf --question "What is the total?"
"""

# ruff: noqa: E402, I001

from __future__ import annotations

import sys
from pathlib import Path

# Make 'invoice_iq' importable when run as a plain script (src/ layout).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from invoice_iq.demo import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
