#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
import os
from src.utils.io import read_yaml, read_jsonl, write_jsonl
from src.preprocess.clean import process_record
import nltk
for pkg in ["stopwords","wordnet"]:
    try: nltk.data.find(f"corpora/{pkg}")
    except LookupError: nltk.download(pkg)

base = read_yaml("configs/base.yaml")
paths = base["paths"]
inp = os.path.join(paths["interim_dir"], "chunks.jsonl")
recs = list(read_jsonl(inp))
extra = base["language"].get("extra_stopwords", []) or base["text_cleaning"].get("extra_stopwords", [])
out = [process_record(r, extra_stop=extra) for r in recs]
outp = os.path.join(paths["processed_dir"], "chunks_tokens.jsonl")
write_jsonl(outp, out)
print(f"Saved tokens to {outp} — {len(out)} records")
PY
