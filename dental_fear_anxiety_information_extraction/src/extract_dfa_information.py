from pathlib import Path
import json

import ollama

client = ollama.Client(
    host="http://localhost:11434",
    timeout=300.0,
)


PASSAGE_DIR = Path("output/pdf_extraction")
OUTPUT_DIR = Path("output/information_extraction")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "qwen2.5:7b"

def remove_empty_values(data):
    return {
        key: value
        for key, value in data.items()
        if value not in (None, [], "")
    }

def has_information(extraction):
    return bool(extraction)


def build_prompt(text):
    return f"""
You are an information extraction system for scientific literature about
Dental Fear and Anxiety (DFA).

Extract all explicitly stated information related to dental fear,
dental anxiety, dental phobia, dentophobia, or closely related
fear/anxiety phenomena in dental contexts.

Group information that describes the same fear/anxiety phenomenon or
context into a coherent record.

For each relevant record extract:

1. fear_anxiety_term
   Exact fear-, anxiety-, or phobia-related terms used in the text.
   Return as a list.

2. event_or_context
   The dental event, treatment, procedure, situation, or context
   associated with the fear/anxiety.

3. who
   The person, patient population, participant group, or other group
   associated with the information.

4. when
   When the fear/anxiety, event, or experience occurs, if explicitly stated.

5. where
   The location or setting, if explicitly stated.

6. cause_or_trigger
   Explicit causes, triggers, antecedents, risk factors, or contributing
   factors associated with the fear/anxiety.
   Return as a list.

7. how_experienced_or_manifested
   How the fear/anxiety is experienced, expressed, or manifested,
   including emotional, cognitive, physiological, and psychomotor
   manifestations.
   Return as a list.

8. response
   Explicit behavioral or other responses to the fear/anxiety.
   Return as a list.

9. outcome_or_impact
   Explicit consequences or impacts of the fear/anxiety, including
   effects on dental attendance, treatment, behavior, health, or
   social functioning.
   Return as a list.

10. other_relevant_information
    Other explicitly stated information important for understanding
    the fear/anxiety phenomenon.
    Return as a list.

11. evidence
    Exact supporting sentence(s) or shortest sufficient text spans
    from the provided text.
    Return as a list.

RULES:

- Use ONLY information explicitly stated in the provided text; do not
  infer or use outside knowledge.
- Preserve the terminology used in the source.
- Omit any field whose value is missing or null.
- Keep [] for list fields when no relevant information is present.
- Do not duplicate the same information across fields.
- Group information describing the same phenomenon into one record;
  create separate records only for clearly distinct phenomena,
  populations, events, or contexts.
- fear_anxiety_term must contain ONLY explicit fear-, anxiety-, or
  phobia-related terms (e.g., dental fear, dental anxiety, dental
  phobia, dentophobia, fear, anxiety). Do not include symptoms,
  behaviors, causes, triggers, procedures, or outcomes.
- event_or_context must capture an explicitly stated dental treatment,
  procedure, visit, situation, or experience (e.g., dental treatment,
  injection, local anesthesia). Do not leave it null when such a
  context is explicitly stated.
- who must contain only explicitly stated persons or populations
  (e.g., patients, children, adults, participants, people).
- Put causes, triggers, antecedents, and contributing factors under
  cause_or_trigger.
- Put emotional, cognitive, physiological, and psychomotor signs or
  symptoms under how_experienced_or_manifested.
- Put actions or behaviors resulting from fear/anxiety under response.
- Put consequences or impacts under outcome_or_impact, including
  treatment delay/avoidance, irregular attendance, poorer oral health,
  impaired quality of life, or treatment difficulties when stated.
- Use other_relevant_information only when the information does not
  clearly belong in another field.
- Evidence must be copied directly from the provided text and contain
  the actual supporting sentence or sufficient text span. Never use
  citation markers (e.g., [1], [3,4], [9-11]), page numbers, author
  information, affiliations, headers, footers, or bibliographic
  metadata as evidence.
- If relevant information is extracted and supporting text is present,
  evidence must not be empty.
- If a passage continues a list or description without explicitly
  repeating the fear/anxiety term, do not infer the term; leave
  fear_anxiety_term empty and extract only the explicitly stated
  information.
- If no relevant DFA information exists, return an empty list.
- Return valid JSON only, with no Markdown or explanations.
- event_or_context: extract the explicitly stated dental event, treatment,
  procedure, visit, or situation associated with the information.
- evidence: copy the actual supporting text from the passage. Never return
  citation markers such as [1], [3,4], or [9-11] as evidence.

Return:

{{
  "extractions": [
    {{
      "fear_anxiety_term": [],
      "event_or_context": null,
      "who": null,
      "when": null,
      "where": null,
      "cause_or_trigger": [],
      "how_experienced_or_manifested": [],
      "response": [],
      "outcome_or_impact": [],
      "other_relevant_information": [],
      "evidence": []
    }}
  ]
}}

TEXT:

{text}
"""


def extract_from_passage(text):
    prompt = build_prompt(text)

    response = client.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        format="json",
        options={
            "temperature": 0,
            "num_predict": 2048,
        },
    )

    content = response["message"]["content"]

    return json.loads(content)


def process_passage_files():
    passage_files = sorted(PASSAGE_DIR.glob("*_passages.json"))

    if not passage_files:
        print("No passage files found in output/")
        return

    for passage_file in passage_files:
        print(f"\nProcessing: {passage_file.name}")

        data = json.loads(
            passage_file.read_text(encoding="utf-8")
        )

        all_extractions = []

        for passage in data["passages"]:
            passage_id = passage["passage_id"]

            print(
                f"  Extracting passage "
                f"{passage_id}/{len(data['passages'])}..."
            )

            try:
                result = extract_from_passage(passage["text"])

                for extraction in result.get("extractions", []):
                  extraction = remove_empty_values(extraction)

                  if not has_information(extraction):
                    continue

                  extraction["passage_id"] = passage_id
                  all_extractions.append(extraction)

            except Exception as error:
                print(
                    f"  Error in passage "
                    f"{passage_id}: {error}"
                )

        output_path = OUTPUT_DIR / (
            passage_file.stem.replace("_passages", "")
            + "_dfa_extractions.json"
        )

        result = {
            "source": data["source"],
            "total_passages": len(data["passages"]),
            "total_extractions": len(all_extractions),
            "extractions": all_extractions,
        }

        output_path.write_text(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print(f"\nTotal extractions: {len(all_extractions)}")
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    process_passage_files()


