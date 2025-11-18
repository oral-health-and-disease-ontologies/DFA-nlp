import os
from io import StringIO
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage

def _extract_page_text(pdf_path, page_numbers=None):
    output = StringIO()
    laparams = LAParams()
    with open(pdf_path, "rb") as f:
        extract_text_to_fp(f, output, laparams=laparams, page_numbers=page_numbers, output_type="text")
    return output.getvalue()

def pdf_to_pages(path):
    pages = []
    with open(path, "rb") as f:
        for i, _ in enumerate(PDFPage.get_pages(f), start=0):
            txt = _extract_page_text(path, page_numbers=[i]) or ""
            pages.append({"page": i + 1, "text": txt})
    title = os.path.basename(path)
    return {"title": title, "pages": pages}

def extract_all(raw_dir: str):
    results = []
    for fname in sorted(os.listdir(raw_dir)):
        if not fname.lower().endswith(".pdf"):
            continue
        full = os.path.join(raw_dir, fname)
        doc = pdf_to_pages(full)
        results.append({"pdf_id": os.path.splitext(fname)[0], **doc})
        print(f"[INFO] Extracted {len(doc['pages'])} pages from {fname}")
    return results
