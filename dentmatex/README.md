# DentMatEx

DentMatEx is a structured information extraction pipeline for extracting dental material information from scientific literature using a locally deployed large language model.

The pipeline converts PDF or DOCX scientific articles into passages, extracts structured dental material information using Qwen2.5-7B through Ollama, and generates RDF knowledge graphs from the extracted information. Each article is processed separately, while an additional cumulative knowledge graph combines information from all processed articles.

## Requirements

- Python 3
- Ollama
- Qwen2.5-7B

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Pull the Qwen model:

```bash
ollama pull qwen2.5:7b
```

## Running DentMatEx

### 1. Convert DOCX to passages

Place PDF or DOCX scientific articles in the `input/` directory and run:

```bash
python scripts/article_to_dentmatex.py
```
The script automatically detects PDF and DOCX articles in the `input/` directory and creates a separate output directory for each article.

Outputs:

```text
outputs/<article_name>/dentmatex_input.jsonl
outputs/<article_name>/dentmatex_input.csv
```

Articles whose passage files already exist are skipped automatically.

### 2. Extract dental material information

```bash
python scripts/dentmatex.py
```

DentMatEx processes the passage files and generates structured extraction results separately for each article.

Outputs:

```text
outputs/<article_name>/dentmatex_results.json
outputs/<article_name>/dentmatex_results.csv
```

DentMatEx extracts material names, material types, brand names, components, properties, property values, units, experimental conditions, applications, and explicit relations. Fields not present in the source passage are omitted from the final extraction.

Articles that already contain completed extraction results are skipped automatically.

### 3. Generate the knowledge graph

```bash
python scripts/build_dentmatex_kg.py
```

A separate RDF knowledge graph is generated for each processed article.

Output:

```text
outputs/<article_name>/dentmatex_knowledge_graph.ttl
```

Article-specific identifiers are included in passage and extraction URIs so that information originating from different articles remains distinguishable when the graphs are combined.

Exisitng article knowledge graphs are skipped automatically.

### 4. Generate the cumulative knowledge graph

Run:

```bash
python scripts/build_cumulative_kg.py
```

The script combines all individual article knowledge graphs into a cumulative RDF knowledge graph.

Output:

```text
outputs/dentmatex_cumulative_knowledge_graph.ttl
```

The cumulative knowledge graph contains the unique RDF from all processed articles while retaining the individual article knowledge graphs separately.

The knowledge graphs are serialized in RDF/Turtle format and can be imported into RDF graph systems such as GraphDB.

## Adding New Articles

To process new literature, add new PDF or DOCX articles to the `input/` directory and run:

```bash
python scripts/article_to_dentmatex.py
python scritps/dentmatex.py
python scripts/build_dentmatex_kg.py
python scripts/build_cumulative_kg.py
```

Previously processed articles are skipped automatically during passage generation, information extraction, and individual knowledge graph generation. Therefore, only newly added articles are processed.
