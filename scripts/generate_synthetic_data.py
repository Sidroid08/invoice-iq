"""CLI: generate a labelled corpus of synthetic invoice/receipt/contract PDFs.

Thin wrapper over `invoice_iq.synthetic.generate_corpus`. Writes PDFs under
`data/generated/<type>/` plus a `labels.json` manifest used by the Phase 3
classifier. The corpus is gitignored (regenerate any time).

Usage:
    python scripts/generate_synthetic_data.py --n 30 --out data/generated --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make 'invoice_iq' importable when run as a plain script (src/ layout).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from invoice_iq.synthetic import generate_corpus  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=20, help="documents per class")
    parser.add_argument("--out", type=Path, default=Path("data/generated"), help="output directory")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    args = parser.parse_args(argv)

    manifest = generate_corpus(args.out, n_per_class=args.n, seed=args.seed)
    labels_path = Path(args.out) / "labels.json"
    labels_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[generate] wrote {len(manifest)} PDFs across 3 classes to {args.out}")
    print(f"[generate] manifest: {labels_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
