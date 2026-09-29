import json
import re
import sys
from pathlib import Path

from rdflib import Graph, Namespace, RDF, RDFS, Literal, URIRef
from rdflib.namespace import XSD


# ============================================================
# NAMESPACES
# ============================================================

EX = Namespace("http://example.org/dentmatex/")
MAT = Namespace("http://example.org/material/")
PROP = Namespace("http://example.org/property/")
TYPE = Namespace("http://example.org/material-type/")
PASSAGE = Namespace("http://example.org/passage/")
EXTRACTION = Namespace("http://example.org/extraction/")


# ============================================================
# NORMALIZATION
# ============================================================

def clean_text(value):
    """
    Clean a text value.
    """

    if value is None:
        return ""

    if not isinstance(value, str):
        value = str(value)

    value = re.sub(r"\s+", " ", value)

    return value.strip()


def make_identifier(text):
    """
    Convert text into a URI-safe identifier.

    Example:

        glass-ionomer cement

    becomes:

        glass_ionomer_cement
    """

    text = clean_text(text).lower()

    text = text.replace("–", "-")
    text = text.replace("—", "-")

    text = re.sub(
        r"[^a-z0-9]+",
        "_",
        text
    )

    text = text.strip("_")

    return text


# ============================================================
# CREATE GRAPH
# ============================================================

def create_graph():

    graph = Graph()

    graph.bind("ex", EX)
    graph.bind("mat", MAT)
    graph.bind("prop", PROP)
    graph.bind("mtype", TYPE)
    graph.bind("passage", PASSAGE)
    graph.bind("extract", EXTRACTION)
    graph.bind("rdf", RDF)
    graph.bind("rdfs", RDFS)

    return graph


# ============================================================
# DEFINE SCHEMA
# ============================================================

def add_schema(graph):

    # --------------------------------------------------------
    # CLASSES
    # --------------------------------------------------------

    classes = {
        EX.Passage: "Passage",
        EX.MaterialExtraction: "Material Extraction",
        EX.Material: "Material",
        EX.MaterialType: "Material Type",
        EX.MaterialProperty: "Material Property"
    }

    for class_uri, label in classes.items():

        graph.add(
            (
                class_uri,
                RDF.type,
                RDFS.Class
            )
        )

        graph.add(
            (
                class_uri,
                RDFS.label,
                Literal(label)
            )
        )

    # --------------------------------------------------------
    # RELATIONSHIPS
    # --------------------------------------------------------

    properties = {
        EX.hasExtraction: "has extraction",
        EX.hasMaterial: "has material",
        EX.hasMaterialType: "has material type",
        EX.hasProperty: "has property"
    }

    for property_uri, label in properties.items():

        graph.add(
            (
                property_uri,
                RDF.type,
                RDF.Property
            )
        )

        graph.add(
            (
                property_uri,
                RDFS.label,
                Literal(label)
            )
        )


# ============================================================
# MATERIAL NODE
# ============================================================

def get_material_node(graph, material_name):

    material_name = clean_text(material_name)

    if not material_name:
        return None

    identifier = make_identifier(
        material_name
    )

    material_uri = MAT[identifier]

    graph.add(
        (
            material_uri,
            RDF.type,
            EX.Material
        )
    )

    graph.add(
        (
            material_uri,
            RDFS.label,
            Literal(material_name)
        )
    )

    graph.add(
        (
            material_uri,
            EX.materialName,
            Literal(material_name)
        )
    )

    return material_uri


# ============================================================
# MATERIAL TYPE NODE
# ============================================================

def get_material_type_node(
    graph,
    material_type
):

    material_type = clean_text(
        material_type
    )

    if not material_type:
        return None

    identifier = make_identifier(
        material_type
    )

    type_uri = TYPE[identifier]

    graph.add(
        (
            type_uri,
            RDF.type,
            EX.MaterialType
        )
    )

    graph.add(
        (
            type_uri,
            RDFS.label,
            Literal(material_type)
        )
    )

    graph.add(
        (
            type_uri,
            EX.typeName,
            Literal(material_type)
        )
    )

    return type_uri


# ============================================================
# PROPERTY NODE
# ============================================================

def get_property_node(
    graph,
    property_name
):

    property_name = clean_text(
        property_name
    )

    if not property_name:
        return None

    identifier = make_identifier(
        property_name
    )

    property_uri = PROP[identifier]

    graph.add(
        (
            property_uri,
            RDF.type,
            EX.MaterialProperty
        )
    )

    graph.add(
        (
            property_uri,
            RDFS.label,
            Literal(property_name)
        )
    )

    graph.add(
        (
            property_uri,
            EX.propertyName,
            Literal(property_name)
        )
    )

    return property_uri


# ============================================================
# OPTIONAL LITERAL FIELDS
# ============================================================

def add_optional_literal(
    graph,
    subject,
    predicate,
    value
):

    value = clean_text(value)

    if value:

        graph.add(
            (
                subject,
                predicate,
                Literal(value)
            )
        )


# ============================================================
# PROCESS ONE EXTRACTION
# ============================================================

def process_extraction(
    graph,
    passage_uri,
    passage_id,
    extraction_data,
    extraction_index,
    article_name
):

    extraction_uri = EXTRACTION[
        f"{article_name}/"
        f"passage_{passage_id}_extraction_{extraction_index}"
    ]

    # --------------------------------------------------------
    # Extraction node
    # --------------------------------------------------------

    graph.add(
        (
            extraction_uri,
            RDF.type,
            EX.MaterialExtraction
        )
    )

    graph.add(
        (
            extraction_uri,
            RDFS.label,
            Literal(
                f"Extraction {passage_id}.{extraction_index}"
            )
        )
    )

    # Passage -> Extraction

    graph.add(
        (
            passage_uri,
            EX.hasExtraction,
            extraction_uri
        )
    )

    # --------------------------------------------------------
    # MATERIAL
    # --------------------------------------------------------

    material_name = extraction_data.get(
        "material_name",
        ""
    )

    material_uri = get_material_node(
        graph,
        material_name
    )

    if material_uri:

        graph.add(
            (
                extraction_uri,
                EX.hasMaterial,
                material_uri
            )
        )

    # --------------------------------------------------------
    # MATERIAL TYPE
    # --------------------------------------------------------

    material_type = extraction_data.get(
        "material_type",
        ""
    )

    type_uri = get_material_type_node(
        graph,
        material_type
    )

    if material_uri and type_uri:

        graph.add(
            (
                material_uri,
                EX.hasMaterialType,
                type_uri
            )
        )

    # --------------------------------------------------------
    # PROPERTY
    # --------------------------------------------------------

    property_name = extraction_data.get(
        "property",
        ""
    )

    property_uri = get_property_node(
        graph,
        property_name
    )

    if property_uri:

        graph.add(
            (
                extraction_uri,
                EX.hasProperty,
                property_uri
            )
        )

    # --------------------------------------------------------
    # OTHER EXTRACTION VALUES
    # --------------------------------------------------------

    add_optional_literal(
        graph,
        extraction_uri,
        EX.propertyValue,
        extraction_data.get(
            "property_value",
            ""
        )
    )

    add_optional_literal(
        graph,
        extraction_uri,
        EX.unit,
        extraction_data.get(
            "unit",
            ""
        )
    )

    add_optional_literal(
        graph,
        extraction_uri,
        EX.condition,
        extraction_data.get(
            "condition",
            ""
        )
    )

    add_optional_literal(
        graph,
        extraction_uri,
        EX.component,
        extraction_data.get(
            "component",
            ""
        )
    )

    add_optional_literal(
        graph,
        extraction_uri,
        EX.application,
        extraction_data.get(
            "application",
            ""
        )
    )

    add_optional_literal(
        graph,
        extraction_uri,
        EX.brandName,
        extraction_data.get(
            "brand_name",
            ""
        )
    )

    add_optional_literal(
        graph,
        extraction_uri,
        EX.relation,
        extraction_data.get(
            "relation",
            ""
        )
    )


# ============================================================
# PROCESS ONE PASSAGE
# ============================================================

def process_passage(
    graph,
    passage_data,
    article_name
):

    passage_id = passage_data.get(
        "passage_id"
    )

    if passage_id is None:
        return

    passage_uri = PASSAGE[
    f"{article_name}/passage_{passage_id}"
    ]

    # --------------------------------------------------------
    # Passage node
    # --------------------------------------------------------

    graph.add(
        (
            passage_uri,
            RDF.type,
            EX.Passage
        )
    )

    graph.add(
        (
            passage_uri,
            RDFS.label,
            Literal(
                f"Passage {passage_id}"
            )
        )
    )

    graph.add(
        (
            passage_uri,
            EX.passageId,
            Literal(
                passage_id,
                datatype=XSD.integer
            )
        )
    )

    # --------------------------------------------------------
    # Original prompt
    # --------------------------------------------------------

    prompt = clean_text(
        passage_data.get(
            "prompt",
            ""
        )
    )

    if prompt:

        graph.add(
            (
                passage_uri,
                EX.prompt,
                Literal(prompt)
            )
        )

    # --------------------------------------------------------
    # Extraction array
    # --------------------------------------------------------

    extractions = passage_data.get(
        "extraction",
        []
    )

    if not isinstance(
        extractions,
        list
    ):

        return

    for extraction_index, extraction_data in enumerate(
        extractions,
        start=1
    ):

        if not isinstance(
            extraction_data,
            dict
        ):
            continue

        process_extraction(
            graph,
            passage_uri,
            passage_id,
            extraction_data,
            extraction_index,
            article_name
        )


# ============================================================
# READ JSON
# ============================================================

def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    # --------------------------------------------------------
    # JSON may be:
    #
    # [
    #   {...},
    #   {...}
    # ]
    #
    # OR
    #
    # {
    #   "results": [...]
    # }
    # --------------------------------------------------------

    if isinstance(data, list):
        return data

    if isinstance(data, dict):

        if "results" in data:

            return data["results"]

        return [data]

    raise ValueError(
        "Unsupported JSON structure."
    )


# ============================================================
# BUILD KNOWLEDGE GRAPH
# ============================================================

def build_knowledge_graph(
    input_json,
    output_ttl
):

    input_json = Path(input_json)
    article_name = make_identifier(input_json.parent.name)

    data = load_json(
        input_json
    )

    graph = create_graph()

    add_schema(
        graph
    )

    for passage_data in data:

        if not isinstance(
            passage_data,
            dict
        ):
            continue

        process_passage(
            graph,
            passage_data,
            article_name
        )

    graph.serialize(
        destination=output_ttl,
        format="turtle"
    )

    print()
    print(
        f"Passages processed: {len(data)}"
    )

    print(
        f"RDF triples created: {len(graph)}"
    )

    print(
        f"Knowledge graph saved to: {output_ttl}"
    )


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    project_root = Path(__file__).resolve().parent.parent
    outputs_dir = project_root / "outputs"

    input_files = sorted(
        outputs_dir.glob("*/dentmatex_results.json")
    )

    if not input_files:
        print("No DentMatEx result files found.")
        sys.exit(0)

    print(f"Found {len(input_files)} article(s).")

    built = 0
    skipped = 0

    for input_path in input_files:

        article_name = input_path.parent.name

        output_path = (
            input_path.parent /
            "dentmatex_knowledge_graph.ttl"
        )

        print()
        print(f"Article: {article_name}")

        # Skip if the knowledge graph already exists
        if output_path.exists():
            print(
                f"Skipping existing knowledge graph: "
                f"{article_name}"
            )
            skipped += 1
            continue

        build_knowledge_graph(
            input_path,
            output_path
        )

        built += 1

    print()
    print("Knowledge graph generation complete.")
    print(f"New graphs created: {built}")
    print(f"Already existing:   {skipped}")