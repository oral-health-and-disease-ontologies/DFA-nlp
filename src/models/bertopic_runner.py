import os, json
from ..utils.io import read_yaml, read_jsonl

def run(cfg_path="configs/bertopic.yaml", base_cfg="configs/base.yaml"):
    try:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer  # ensure present
        import umap, hdbscan  # noqa
    except Exception:
        print("[INFO] BERTopic deps not installed — skipping BERTopic.")
        return None

    base = read_yaml(base_cfg); cfg = read_yaml(cfg_path)
    proc_path = os.path.join(base["paths"]["processed_dir"], "chunks_tokens.jsonl")
    if not os.path.exists(proc_path):
        print("[WARN] Missing processed chunks. Run scripts/02_preprocess.sh first.")
        return None

    records = list(read_jsonl(proc_path))
    docs = [" ".join(r.get("tokens", [])) for r in records]

    topic_model = BERTopic(
        embedding_model=cfg.get("embedding_model","sentence-transformers/all-MiniLM-L6-v2"),
        min_topic_size=int(cfg.get("min_topic_size",3)),
        nr_topics=cfg.get("nr_topics","auto"),
        top_n_words=int(cfg.get("top_n_words",15)),
    )
    topics, _ = topic_model.fit_transform(docs)

    out_dir = os.path.join(base["paths"]["models_dir"], "bertopic")
    os.makedirs(out_dir, exist_ok=True)

    # Save model to a FILE path (pickle) instead of the directory
    model_path = os.path.join(out_dir, "bertopic_model.pkl")
    topic_model.save(model_path)

    # Export top terms
    info = []
    for tid in sorted(set(t for t in topics if t != -1)):
        terms = [w for w,_ in topic_model.get_topic(tid)]
        info.append({"topic": int(tid), "terms": terms})
    json.dump(info, open(os.path.join(out_dir, "topics.json"),"w",encoding="utf-8"), indent=2)

    print("BERTopic written to", out_dir)
    return out_dir

if __name__ == "__main__":
    run()
