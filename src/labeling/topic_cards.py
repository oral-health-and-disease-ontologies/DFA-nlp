import os, json
from ..utils.io import read_yaml

def _write_card(dirpath, model_name, idx, terms):
    os.makedirs(dirpath, exist_ok=True)
    path = os.path.join(dirpath, f"topic_{idx:02d}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {model_name.upper()} — Topic {idx}\n\n")
        f.write("Top terms:\n\n")
        for t in terms:
            f.write(f"- {t}\n")
    return path

def main():
    base = read_yaml("configs/base.yaml")
    reports = os.path.join(base["paths"]["reports_dir"], "topic_cards")
    nmf = os.path.join(base["paths"]["models_dir"], "nmf", "best_terms.json")
    if os.path.exists(nmf):
        data = json.load(open(nmf,"r",encoding="utf-8"))
        for i, terms in enumerate(data.get("terms", [])):
            _write_card(os.path.join(reports,"nmf"), "nmf", i, terms)
    corex = os.path.join(base["paths"]["models_dir"], "corex", "topics.json")
    if os.path.exists(corex):
        data = json.load(open(corex,"r",encoding="utf-8"))
        for row in data:
            _write_card(os.path.join(reports,"corex"), "corex", row["topic"], row["terms"])
    bt = os.path.join(base["paths"]["models_dir"], "bertopic", "topics.json")
    if os.path.exists(bt):
        data = json.load(open(bt,"r",encoding="utf-8"))
        for row in data:
            _write_card(os.path.join(reports,"bertopic"), "bertopic", row["topic"], row["terms"])
    print("Cards written to reports/topic_cards/*")

if __name__ == "__main__":
    main()
