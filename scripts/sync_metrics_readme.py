"""Sync trained-classifier metrics into the README.

Reads ``models/metrics.json`` (produced by the Phase 3 training run) and replaces
the block between the ``<!-- METRICS:START -->`` and ``<!-- METRICS:END -->``
markers in README.md with a formatted summary (accuracy, macro-F1, confusion
matrix). This keeps the recruiter-facing README's numbers honest and current.

Safe to run any time: if metrics.json is absent (e.g. before Phase 3), it leaves
a "not yet trained" placeholder and exits 0 — so it never breaks CI.

Usage:
    python scripts/sync_metrics_readme.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METRICS_PATH = ROOT / "models" / "metrics.json"
README_PATH = ROOT / "README.md"

START = "<!-- METRICS:START -->"
END = "<!-- METRICS:END -->"


def _render_placeholder() -> str:
    return (
        "_Classifier not trained yet. Run the Phase 3 training "
        "(`python -m invoice_iq.classifier.train`) to populate this section._"
    )


def _render_metrics(metrics: dict[str, object]) -> str:
    """Render a metrics dict into a Markdown summary.

    Expected (Phase 3) shape (extra keys are ignored, missing keys tolerated):
        {
          "accuracy": 0.97,
          "macro_f1": 0.96,
          "labels": ["invoice", "receipt", "contract"],
          "confusion_matrix": [[..],[..],[..]],
          "n_test": 120,
          "trained_at": "2026-06-04T12:00:00Z"
        }
    """
    lines: list[str] = []
    acc = metrics.get("accuracy")
    f1 = metrics.get("macro_f1")
    n_test = metrics.get("n_test")
    trained_at = metrics.get("trained_at")

    headline_bits = []
    if isinstance(acc, (int, float)):
        headline_bits.append(f"**Accuracy:** {acc:.1%}")
    if isinstance(f1, (int, float)):
        headline_bits.append(f"**Macro-F1:** {f1:.3f}")
    if isinstance(n_test, int):
        headline_bits.append(f"**Test samples:** {n_test}")
    if headline_bits:
        lines.append(" &nbsp;|&nbsp; ".join(headline_bits))
        lines.append("")

    labels = metrics.get("labels")
    cm = metrics.get("confusion_matrix")
    if isinstance(labels, list) and isinstance(cm, list):
        header = "| actual \\ pred | " + " | ".join(str(x) for x in labels) + " |"
        sep = "|" + "---|" * (len(labels) + 1)
        lines.append(header)
        lines.append(sep)
        for label, row in zip(labels, cm):
            cells = " | ".join(str(v) for v in row)  # type: ignore[union-attr]
            lines.append(f"| **{label}** | {cells} |")
        lines.append("")

    if isinstance(trained_at, str):
        lines.append(f"_Last trained: {trained_at}_")

    return "\n".join(lines).strip() or _render_placeholder()


def main() -> int:
    if not README_PATH.exists():
        print(f"[sync-metrics] {README_PATH} not found; nothing to do.")
        return 0

    readme = README_PATH.read_text(encoding="utf-8")
    if START not in readme or END not in readme:
        print(
            "[sync-metrics] markers not found in README "
            f"({START!r} / {END!r}); skipping."
        )
        return 0

    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        body = _render_metrics(metrics)
        print("[sync-metrics] injected metrics from models/metrics.json")
    else:
        body = _render_placeholder()
        print("[sync-metrics] no metrics.json yet; wrote placeholder")

    pre = readme.split(START)[0]
    post = readme.split(END)[1]
    new_readme = f"{pre}{START}\n{body}\n{END}{post}"

    if new_readme != readme:
        README_PATH.write_text(new_readme, encoding="utf-8")
        print("[sync-metrics] README.md updated")
    else:
        print("[sync-metrics] README.md already up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
