#!/usr/bin/env bash
set -euo pipefail
python -m src.eval.extrinsic || true
python -m src.eval.coherence || true
python -m src.eval.compare || true
