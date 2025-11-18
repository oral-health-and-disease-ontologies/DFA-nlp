#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
import os, re
from src.utils.io import read_yaml, write_jsonl
from src.ingest.pdf_to_text import extract_all
from src.ingest.chunker import chunk_pages

base = read_yaml("configs/base.yaml"); paths = base["paths"]
docs = extract_all(paths["raw_dir"])

REF_HEAD = re.compile(r"^\s*(references|bibliography)\b", re.I)
DOI_OR_URL = re.compile(r"\b(doi:|https?://|www\.)", re.I)
PUBLISHER_FOOT = re.compile(r"(sagepub|copyright|all rights reserved)", re.I)

records = []
for d in docs:
    chunks = chunk_pages(d, target_tokens=base["chunking"]["target_tokens"],
                         overlap_tokens=base["chunking"]["overlap_tokens"])
    for ch in chunks:
        text = ch["text"] or ""
        if REF_HEAD.search(text) or len(DOI_OR_URL.findall(text)) >= 2:
            continue
        if len(text.split()) < 25 and PUBLISHER_FOOT.search(text):
            continue
        records.append({"pdf_id": d["pdf_id"], "title": d["title"],
                        "start_page": ch["start_page"], "end_page": ch["end_page"],
                        "text": text})
outp = os.path.join(paths["interim_dir"], "chunks.jsonl")
write_jsonl(outp, records)
print(f"Wrote {len(records)} chunks to {outp}")
PY
