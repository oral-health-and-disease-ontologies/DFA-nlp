import csv
import json
import sys
from pathlib import Path

from docx import Document
from pypdf import PdfReader


def read_docx(path):
    """Extract non-empty paragraphs from a DOCX article."""

    doc = Document(path)
    passages = []

    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()

        if text:
            passages.append(text)

    return passages


def read_pdf(path):
    """Extract paragraph-like passages from a PDF article."""

    reader = PdfReader(path)
    passages = []

    for page in reader.pages:
        text = page.extract_text()

        if not text:
            continue

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        current_passage = ""

        for line in lines:

            # Repair words split across PDF lines:
            # "glass-" + "ionomer" -> "glass-ionomer"
            if current_passage.endswith("-"):
                current_passage += line

            elif current_passage:
                current_passage += " " + line

            else:
                current_passage = line

            # End passage when the accumulated text
            # appears to end a sentence.
            if line.endswith((".", "?", "!")):
                passages.append(current_passage.strip())
                current_passage = ""

        # Keep remaining text at end of page.
        if current_passage:
            passages.append(current_passage.strip())

    return passages


def read_article(path):
    """Read a PDF or DOCX article."""

    path = Path(path)

    suffix = path.suffix.lower()

    if suffix == ".docx":
        return read_docx(path)

    if suffix == ".pdf":
        return read_pdf(path)

    raise ValueError(
        "Unsupported file type. "
        "DentMatEx accepts .pdf and .docx files."
    )



def convert_to_dentmatex(article_path):
    article_path = Path(article_path)

    if not article_path.exists():
        raise FileNotFoundError(
            f"Article not found: {article_path}"
        )

    project_root = Path(__file__).resolve().parent.parent

    article_name = article_path.stem

    output_dir = project_root / "outputs" / article_name

    jsonl_output_path = (
        output_dir / "dentmatex_input.jsonl"
    )

    csv_output_path = (
        output_dir / "dentmatex_input.csv"
    )

    # Skip article if passage files already exist
    if jsonl_output_path.exists() and csv_output_path.exists():
        print(f"Skipping already processed: {article_path.name}")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    passages = read_article(article_path)


    # ---------------------------------
    # Write DentMatEx JSONL
    # ---------------------------------

    with open(
        jsonl_output_path,
        "w",
        encoding="utf-8"
    ) as out:

        for i, passage in enumerate(
            passages,
            start=1
        ):
            record = {
                "passage_id": i,
                "prompt": passage + "\n\n###\n\n"
            }

            out.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

    # ---------------------------------
    # Write human-readable CSV
    # ---------------------------------

    with open(
        csv_output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as out:

        writer = csv.DictWriter(
            out,
            fieldnames=[
                "passage_id",
                "text"
            ]
        )

        writer.writeheader()

        for i, passage in enumerate(
            passages,
            start=1
        ):
            writer.writerow({
                "passage_id": i,
                "text": passage
            })

    print()
    print(f"Input article: {article_path}")
    print(f"Extracted {len(passages)} passages")

    print(
        f"Saved DentMatEx JSONL input to: "
        f"{jsonl_output_path}"
    )

    print(
        f"Saved readable CSV to: "
        f"{csv_output_path}"
    )

    return True

if __name__ == "__main__":

    project_root = Path(__file__).resolve().parent.parent
    input_dir = project_root / "input"

    articles = sorted(
        path
        for path in input_dir.iterdir()
        if path.suffix.lower() in {".pdf", ".docx"}
    )

    if not articles:
        print("No PDF or DOCX articles found.")
        sys.exit(0)

    processed = 0
    skipped = 0

    for article in articles:

        if convert_to_dentmatex(article):
            processed += 1
        else:
            skipped += 1

    print()
    print("Complete")
    print(f"New articles processed: {processed}")
    print(f"Already processed:      {skipped}")