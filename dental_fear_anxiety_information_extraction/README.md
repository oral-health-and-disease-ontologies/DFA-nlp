# Dental Fear and Anxiety Information Extraction

This project extracts structured information about Dental Fear and Anxiety (DFA) from scientific literature.

The pipeline uses PDF articles as input, extracts text passages, and uses a local Qwen2.5 7B model through Ollama to identify information related to dental fear and anxiety and related phenomena.

## Usage

Place PDF articles in input/.

Extract text passages:

```bash
python src/pdf_extractor.py
```

Run DFA information extraction:

```bash
python src/extract_dfa_information.py
```

The resulting structured JSON files are saved in:

```text
output/information_extraction/
```

