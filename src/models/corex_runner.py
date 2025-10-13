import os, json
import numpy as np
from scipy.sparse import csr_matrix
from ..utils.io import read_yaml, read_jsonl
from ..features.tfidf import build_tfidf_strs
from ..features.seeds import load_seeds

def _get_corex():
    try:
        from corex import Corex
        return Corex
    except Exception:
        try:
            from corextopic import corextopic as ct  # fork that supports newer Python
            return ct.Corex
        except Exception:
            return None

def run(cfg_path="configs/corex.yaml", base_cfg="configs/base.yaml"):
    Corex = _get_corex()
    if Corex is None:
        print("[INFO] CorEx not available — skipping.")
        return None

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

    # CorEx expects binary, sparse input; binarize TF-IDF and keep CSR
    X_bin = csr_matrix((X > 0).astype(np.int8))

    seeds = load_seeds(cfg.get("anchors_file","configs/seeds.yaml"))
    anchors = []
    for _, terms in seeds.items():
        idxs = [i for i, v in enumerate(vocab) if v in terms]
        if idxs: anchors.append(idxs)

    model = Corex(n_hidden=int(cfg.get("n_topics",5)), seed=base["random_seed"])
    if cfg.get("use_anchors", True) and anchors:
        model.fit(X_bin, words=vocab, anchors=anchors, anchor_strength=float(cfg.get("anchor_strength",2.0)))
    else:
        model.fit(X_bin, words=vocab)

    topics = []
    try:
        for i, t in enumerate(model.get_topics(n_words=15)):
            terms = [w if isinstance(w, str) else w[0] for w in t]
            topics.append({"topic": i, "terms": terms})
    except Exception:
        # conservative fallback
        for i in range(int(cfg.get("n_topics",5))):
            topics.append({"topic": i, "terms": []})

    out_dir = os.path.join(base["paths"]["models_dir"], "corex")
    os.makedirs(out_dir, exist_ok=True)
    json.dump(topics, open(os.path.join(out_dir, "topics.json"),"w",encoding="utf-8"), indent=2)
    print("CorEx written to", out_dir)
    return out_dir

if __name__ == "__main__":
    run()
