import os, csv, json
from ..utils.io import read_yaml
from ..features.seeds import load_seeds

def _overlap(terms, seeds):
    s = set(terms); return len(s & set(seeds))

def _write_rows(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["model","seed_set","overlap"])
        for r in rows: w.writerow(r)

def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    base = read_yaml("configs/base.yaml")
    seeds = load_seeds("configs/seeds.yaml")
    rows = []

    nmf_path = os.path.join(base["paths"]["models_dir"], "nmf", "best_terms.json")
    if os.path.exists(nmf_path):
        nmf = _read_json(nmf_path)
        for seed_name, seed_terms in seeds.items():
            ov = max(_overlap(t, seed_terms) for t in nmf.get("terms", [])) if nmf.get("terms") else 0
            rows.append(["nmf", seed_name, ov])

    corex_path = os.path.join(base["paths"]["models_dir"], "corex", "topics.json")
    if os.path.exists(corex_path):
        corex = _read_json(corex_path)
        for seed_name, seed_terms in seeds.items():
            ov = max(_overlap(t["terms"], seed_terms) for t in corex) if corex else 0
            rows.append(["corex", seed_name, ov])

    bt_path = os.path.join(base["paths"]["models_dir"], "bertopic", "topics.json")
    if os.path.exists(bt_path):
        bt = _read_json(bt_path)
        for seed_name, seed_terms in seeds.items():
            ov = max(_overlap(t["terms"], seed_terms) for t in bt) if bt else 0
            rows.append(["bertopic", seed_name, ov])

    _write_rows(os.path.join(base["paths"]["evaluation_dir"], "extrinsic_overlap.csv"), rows)
    print("Evaluation updated in evaluation/*.csv")

if __name__ == "__main__":
    main()
