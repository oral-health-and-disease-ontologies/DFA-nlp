import os, json, csv
from ..utils.io import read_yaml
try:
    from gensim.corpora import Dictionary
    from gensim.models.coherencemodel import CoherenceModel
    GENSIM_OK = True
except Exception:
    GENSIM_OK = False

def _load_tokens(proc_path):
    toks = []
    with open(proc_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            toks.append(r.get("tokens", []))
    return toks

def _norm_topic_raw(t):
    # ensure list[str], strip empties
    if not t: return []
    return [str(x).strip() for x in t if str(x).strip()]

def _to_dict_tokens(topic_terms, dct):
    # 1) try underscored whole phrases
    canned = [w.replace(" ", "_") for w in topic_terms]
    keep = [w for w in canned if w in dct.token2id]
    if keep:
        return keep
    # 2) fallback: split into unigrams and keep those present
    parts = []
    for w in canned:
        parts.extend(w.split("_"))
    keep_parts = [p for p in parts if p in dct.token2id]
    return keep_parts


def main():
    base = read_yaml("configs/base.yaml")
    proc = os.path.join(base["paths"]["processed_dir"], "chunks_tokens.jsonl")
    if not os.path.exists(proc) or not GENSIM_OK:
        return
    tokens = _load_tokens(proc)
    # Also underscore the training tokens so phrases may match
    tokens = [[w.replace(" ", "_") for w in doc] for doc in tokens]
    dct = Dictionary(tokens)

    rows = [["model","topic_index","coherence_c_v"]]

    # NMF
    nmf = os.path.join(base["paths"]["models_dir"], "nmf", "best_terms.json")
    if os.path.exists(nmf):
        data = json.load(open(nmf,"r",encoding="utf-8"))
        for i,t in enumerate(data.get("terms", [])):
            topic = _to_dict_tokens(_norm_topic_raw(t), dct)
            if not topic: continue
            cm = CoherenceModel(topics=[topic], texts=tokens, dictionary=dct, coherence="c_v")
            rows.append(["nmf", i, float(cm.get_coherence())])

    # CorEx
    corex = os.path.join(base["paths"]["models_dir"], "corex", "topics.json")
    if os.path.exists(corex):
        data = json.load(open(corex,"r",encoding="utf-8"))
        for row in data:
            topic = _to_dict_tokens(_norm_topic_raw(row.get("terms", [])), dct)
            if not topic: continue
            cm = CoherenceModel(topics=[topic], texts=tokens, dictionary=dct, coherence="c_v")
            rows.append(["corex", int(row["topic"]), float(cm.get_coherence())])

    # BERTopic
    bt = os.path.join(base["paths"]["models_dir"], "bertopic", "topics.json")
    if os.path.exists(bt):
        data = json.load(open(bt,"r",encoding="utf-8"))
        for row in data:
            topic = _to_dict_tokens(_norm_topic_raw(row.get("terms", [])), dct)
            if not topic: continue
            cm = CoherenceModel(topics=[topic], texts=tokens, dictionary=dct, coherence="c_v")
            rows.append(["bertopic", int(row["topic"]), float(cm.get_coherence())])

    out = os.path.join(base["paths"]["evaluation_dir"], "coherence.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out,"w",newline="",encoding="utf-8") as f:
        w = csv.writer(f); w.writerows(rows)
    print("Coherence (c_v) written to evaluation/coherence.csv")

if __name__ == "__main__":
    main()
