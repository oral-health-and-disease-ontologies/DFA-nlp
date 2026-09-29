from pathlib import Path

from rdflib import Graph


def build_cumulative_kg():

    project_root = Path(__file__).resolve().parent.parent
    outputs_dir = project_root / "outputs"

    cumulative_path = (
        outputs_dir /
        "dentmatex_cumulative_knowledge_graph.ttl"
    )

    kg_files = sorted(
        outputs_dir.glob(
            "*/dentmatex_knowledge_graph.ttl"
        )
    )

    if not kg_files:
        print("No article knowledge graphs found.")
        return

    graph = Graph()

    print(f"Found {len(kg_files)} article knowledge graph(s).")

    for kg_file in kg_files:

        print(f"Adding: {kg_file.parent.name}")

        graph.parse(
            kg_file,
            format="turtle"
        )

    graph.serialize(
        destination=cumulative_path,
        format="turtle"
    )

    print()
    print(
        f"Cumulative RDF triples: {len(graph)}"
    )
    print(
        f"Cumulative knowledge graph saved to: "
        f"{cumulative_path}"
    )


if __name__ == "__main__":
    build_cumulative_kg()