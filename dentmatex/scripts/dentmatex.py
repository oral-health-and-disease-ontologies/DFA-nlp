import csv
import json
import sys
from pathlib import Path

from ollama import chat


MODEL = "qwen2.5:7b"


DentMatEx_SCHEMA = {
    "type": "object",
    "properties": {
        "materials": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "material_name": {"type": "string"},
                    "material_type": {"type": "string"},
                    "brand_name": {"type": "string"},
                    "component": {"type": "string"},
                    "property": {"type": "string"},
                    "property_value": {"type": "string"},
                    "unit": {"type": "string"},
                    "condition": {"type": "string"},
                    "application": {"type": "string"},
                    "relation": {"type": "string"}
                },
                "required": [
                    "material_name",
                    "material_type",
                    "brand_name",
                    "component",
                    "property",
                    "property_value",
                    "unit",
                    "condition",
                    "application",
                    "relation"
                ]
            }
        }
    },
    "required": ["materials"]
}


SYSTEM_PROMPT = """
You are performing structured information extraction from scientific literature about dental restorative materials.

Extract only information explicitly stated in the passage. Do not infer or invent missing information.

For each material, extract:

- material_name: name of the material
- material_type: type or class of material
- brand_name: commercial or product name
- component: constituent or component of the material
- property: material property or measured characteristic
- property_value: numerical or textual value of the property
- unit: unit associated with the property value
- condition: experimental or environmental condition under which
  the property was measured
- application: stated application or use of the material
- relation: explicit relationship involving the material,
  component, property, condition, or application

Use an empty string "" when information is not explicitly stated.

If multiple distinct materials are described, return a separate record for each material.

If no relevant material information is present, return:

{"materials": []}

Return valid JSON only.
"""


def extract_passage(text):
    response = chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": text
            }
        ],
        format=DentMatEx_SCHEMA,
        options={
            "temperature": 0
        }
    )

    return json.loads(response.message.content)


def clean_material(material):
    """
    Remove fields whose values are empty.

    Examples removed:
        ""
        None
        []
        {}

    Also removes strings containing only spaces.
    """

    cleaned = {}

    for key, value in material.items():

        if value is None:
            continue

        if isinstance(value, str):
            value = value.strip()

            if value == "":
                continue

        if value == [] or value == {}:
            continue

        cleaned[key] = value

    return cleaned


def clean_extractions(materials):
    """
    Clean every extracted material and remove
    completely empty material records.
    """

    cleaned_materials = []

    for material in materials:

        cleaned = clean_material(material)

        if cleaned:
            cleaned_materials.append(cleaned)

    return cleaned_materials


def run_dentmatex(input_path):

    input_path = Path(input_path)


    json_output = (
        input_path.parent /
        "dentmatex_results.json"
    )

    csv_output = (
        input_path.parent /
        "dentmatex_results.csv"
    )

    # Skip articles that have already been processed
    if json_output.exists() and csv_output.exists():
        print(
            f"Skipping already processed article: "
            f"{input_path.parent.name}"
        )
        return

    results = []

    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            record = json.loads(line)

            passage_id = record["passage_id"]

            prompt = record["prompt"].replace(
                "\n\n###\n\n",
                ""
            )

            print(
                f"Processing passage {passage_id}..."
            )

            extraction = extract_passage(prompt)

            # ---------------------------------
            # REMOVE EMPTY FIELDS
            # ---------------------------------

            materials = clean_extractions(
                extraction.get("materials", [])
            )

            results.append({
                "passage_id": passage_id,
                "prompt": prompt,
                "extraction": materials
            })

    # ---------------------------------
    # SAVE CLEAN JSON
    # ---------------------------------

    with open(
        json_output,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2
        )

    # ---------------------------------
    # SAVE CLEAN CSV
    #
    # Only non-empty fields are written.
    # Each field becomes one row.
    # ---------------------------------

    with open(
        csv_output,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "passage_id",
                "extraction_id",
                "field",
                "value"
            ]
        )

        writer.writeheader()

        for result in results:

            for extraction_id, material in enumerate(
                result["extraction"]
            ):

                for field, value in material.items():

                    writer.writerow({
                        "passage_id":
                            result["passage_id"],

                        "extraction_id":
                            extraction_id,

                        "field":
                            field,

                        "value":
                            value
                    })

    print()

    print(
        f"JSON results saved to: {json_output}"
    )

    print(
        f"CSV results saved to:  {csv_output}"
    )


if __name__ == "__main__":

    project_root = Path(__file__).resolve().parent.parent
    outputs_dir = project_root / "outputs"

    input_files = sorted(
        outputs_dir.glob("*/dentmatex_input.jsonl")
    )

    if not input_files:
        print("No DentMatEx input files found.")
        sys.exit(0)

    print(
        f"Found {len(input_files)} article(s)."
    )
    print()

    for input_file in input_files:
        print(
            f"Article: {input_file.parent.name}"
        )

        run_dentmatex(input_file)

        print()

    print("DentMatEx extraction complete.")