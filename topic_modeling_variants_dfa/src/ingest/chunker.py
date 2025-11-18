import re

def _tokenize(txt: str):
    return re.findall(r"\w+|\S", txt)

def chunk_pages(doc, target_tokens=320, overlap_tokens=50):
    chunks = []
    buf, count, start_page = [], 0, None
    for page in doc["pages"]:
        toks = _tokenize(page["text"] or "")
        i = 0
        while i < len(toks):
            if start_page is None:
                start_page = page["page"]
            need = target_tokens - count
            take = min(need, len(toks) - i)
            buf.extend(toks[i:i+take]); i += take; count += take
            if count >= target_tokens:
                chunks.append({"start_page": start_page, "end_page": page["page"], "text": " ".join(buf)})
                buf = buf[-overlap_tokens:]
                count = len(buf)
                start_page = None
    if buf:
        chunks.append({"start_page": start_page or 1, "end_page": doc["pages"][-1]["page"], "text": " ".join(buf)})
    return chunks
