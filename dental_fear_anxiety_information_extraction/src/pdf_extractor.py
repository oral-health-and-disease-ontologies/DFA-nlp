from pathlib import Path
import json
import re

from pypdf import PdfReader


INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output/pdf_extraction")


def extract_text_from_pdf(pdf_path):
    """Extract text from all pages of a PDF."""

    reader = PdfReader(pdf_path)

    pages = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n".join(pages)


def extract_passages(text):
    """Convert extracted article text into passage-like blocks."""

    lines = text.splitlines()

    passages = []
    current = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # Repair words split across PDF line boundaries
        if current and current[-1].endswith("-"):
            current[-1] = current[-1][:-1] + line
        else:
            current.append(line)

        # End passage when the accumulated text ends a sentence
        joined = " ".join(current)

        if re.search(r'[.!?]["\']?$', joined):
            passages.append(joined)
            current = []

    if current:
        passages.append(" ".join(current))

    return passages


def process_pdfs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(INPUT_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in input/")
        return

    print(f"Found {len(pdf_files)} PDF file(s).\n")

    for pdf_path in pdf_files:

        print(f"Processing: {pdf_path.name}")

        text = extract_text_from_pdf(pdf_path)

        # Save raw extracted text
        txt_path = OUTPUT_DIR / f"{pdf_path.stem}.txt"

        txt_path.write_text(
            text,
            encoding="utf-8",
        )

        # Create passages
        passages = extract_passages(text)

        passage_records = []

        for i, passage in enumerate(passages, start=1):
            passage_records.append(
                {
                    "passage_id": i,
                    "text": passage,
                }
            )

        passage_path = OUTPUT_DIR / f"{pdf_path.stem}_passages.json"

        passage_path.write_text(
            json.dumps(
                {
                    "source": pdf_path.name,
                    "total_passages": len(passage_records),
                    "passages": passage_records,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print(f"Characters extracted: {len(text):,}")
        print(f"Passages extracted: {len(passages)}")
        print(f"Saved text: {txt_path}")
        print(f"Saved passages: {passage_path}\n")


if __name__ == "__main__":
    process_pdfs()
