#!/usr/bin/env bash
set -euo pipefail
python -m src.models.nmf_runner || echo "[WARN] NMF failed."
python -m src.models.corex_runner || echo "[INFO] CorEx skipped or failed."
python -m src.models.bertopic_runner || echo "[INFO] BERTopic skipped (deps missing)."
