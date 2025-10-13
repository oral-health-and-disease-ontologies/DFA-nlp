import os, json
from ..utils.io import read_yaml, read_jsonl
from ..features.tfidf import build_tfidf_strs
from sklearn.decomposition import NMF

def top_terms(H, vocab, topn=15):
    return [[vocab[idx] for idx in comp.argsort()[-topn:][::-1]] for comp in H]

def _nmf_kwargs(cfg: dict):
    return dict(
        init=cfg.get("init", "nndsvda"),
        max_iter=int(cfg.get("max_iter", 1000)),
        l1_ratio=float(cfg.get("l1_ratio", 0.5)),
        alpha_W=float(cfg.get("alpha_W", 0.1)),
        alpha_H=float(cfg.get("alpha_H", 0.1)),
    )

def run(cfg_path="configs/nmf.yaml", base_cfg="configs/base.yaml"):
    base = read_yaml(base_cfg); cfg = read_yaml(cfg_path)
    proc_path = os.path.join(base["paths"]["processed_dir"], "chunks_tokens.jsonl")
    if not os.path.exists(proc_path):
        raise FileNotFoundError("Run scripts/02_preprocess.sh first.")
    records = list(read_jsonl(proc_path))
    X, vocab, _ = build_tfidf_strs(
        records,
        ngram=(base["tfidf"]["ngram_min"], base["tfidf"]["ngram_max"]),
        min_df=base["tfidf"]["min_df"],
        max_df=base["tfidf"]["max_df"],
    )
    best = None; results = []
    for k in range(int(cfg["k_min"]), int(cfg["k_max"]) + 1):
        nmf = NMF(n_components=int(k), random_state=base["random_seed"], **_nmf_kwargs(cfg))
        W = nmf.fit_transform(X); H = nmf.components_
        err = float(nmf.reconstruction_err_)
        terms = top_terms(H, vocab, topn=15)
        results.append({"k": int(k), "reconstruction_error": err, "terms": terms})
        if best is None or err < best["reconstruction_error"]:
            best = {"k": int(k), "reconstruction_error": err, "terms": terms}
    out_dir = os.path.join(base["paths"]["models_dir"], "nmf")
    os.makedirs(out_dir, exist_ok=True)
    json.dump(results, open(os.path.join(out_dir, "grid_results.json"),"w",encoding="utf-8"), indent=2)
    json.dump({"k": best["k"], "terms": best["terms"]}, open(os.path.join(out_dir, "best_terms.json"),"w",encoding="utf-8"), indent=2)
    print("NMF written to", out_dir)
    return out_dir

if __name__ == "__main__":
    run()
