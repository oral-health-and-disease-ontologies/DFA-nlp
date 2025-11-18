from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def _docs_from_records(records):
    docs_tokens = [" ".join(r.get("tokens", []) or []) for r in records]
    if any(len(d.strip()) > 0 for d in docs_tokens):
        return docs_tokens, True
    docs_text = [r.get("text", "") for r in records]
    return docs_text, False

def build_tfidf_strs(records, ngram=(1,3), min_df=1, max_df=0.9):
    docs, using_tokens = _docs_from_records(records)
    vec = TfidfVectorizer(
        ngram_range=ngram,
        min_df=min_df,
        max_df=max_df,
        lowercase=True,
        token_pattern=r"(?u)\b[\w\-']+\b",
    )
    X = vec.fit_transform(docs)
    vocab = np.array(vec.get_feature_names_out())
    if X.shape[1] == 0:
        raise ValueError(f"Empty TF–IDF vocabulary (using_tokens={using_tokens}).")
    return X, vocab, vec
