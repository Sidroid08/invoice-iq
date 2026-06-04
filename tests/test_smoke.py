"""Phase 0 smoke test: proves the package imports and the test harness runs."""

import invoice_iq


def test_package_imports() -> None:
    assert invoice_iq.__version__ == "0.1.0"


def test_python_is_311() -> None:
    import sys

    assert sys.version_info[:2] == (3, 11), "project is pinned to Python 3.11"
