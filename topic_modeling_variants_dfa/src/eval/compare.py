import os, pandas as pd, json
from ..utils.io import read_yaml

def main():
    base = read_yaml("configs/base.yaml")
    evdir = base["paths"]["evaluation_dir"]
    rows = []

    # extrinsic (seed overlap)
    extrinsic = os.path.join(evdir, "extrinsic_overlap.csv")
    if os.path.exists(extrinsic):
        df = pd.read_csv(extrinsic)
        agg = df.groupby("model")["overlap"].sum().reset_index().rename(columns={"overlap":"seed_overlap_total"})
        rows.append(agg.set_index("model"))

    # coherence (optional; only if gensim present and file exists)
    coh = os.path.join(evdir, "coherence.csv")
    if os.path.exists(coh):
        df = pd.read_csv(coh)
        agg = df.groupby("model")["coherence_c_v"].mean().reset_index().rename(columns={"coherence_c_v":"coherence_c_v_mean"})
        rows.append(agg.set_index("model"))

    # diversity (unique term share)
    models_dir = base["paths"]["models_dir"]
    div = {}
    # NMF
    p = os.path.join(models_dir,"nmf","best_terms.json")
    if os.path.exists(p):
        data = json.load(open(p,"r",encoding="utf-8"))
        terms = [t for topic in data.get("terms",[]) for t in topic]
        div["nmf"] = len(set(terms))/max(1,len(terms))
    # Corex
    p = os.path.join(models_dir,"corex","topics.json")
    if os.path.exists(p):
        data = json.load(open(p,"r",encoding="utf-8"))
        terms = [t for row in data for t in row.get("terms",[])]
        div["corex"] = len(set(terms))/max(1,len(terms))
    # BERTopic
    p = os.path.join(models_dir,"bertopic","topics.json")
    if os.path.exists(p):
        data = json.load(open(p,"r",encoding="utf-8"))
        terms = [t for row in data for t in row.get("terms",[])]
        div["bertopic"] = len(set(terms))/max(1,len(terms))

    if div:
        rows.append(pd.DataFrame.from_dict(div, orient="index", columns=["term_diversity"]).astype(float))

    if not rows:
        print("No evaluation files present.")
        return

    out = pd.concat(rows, axis=1).sort_index()
    os.makedirs(evdir, exist_ok=True)
    out.to_csv(os.path.join(evdir,"model_comparison.csv"))
    print("Wrote evaluation/model_comparison.csv")

if __name__ == "__main__":
    main()
