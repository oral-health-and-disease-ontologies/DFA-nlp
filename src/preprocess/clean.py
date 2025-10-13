import re, json
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
CITATION_RE = re.compile(
    r"\(("
    r"(?:[a-z][a-z\-]+(?:\s*&\s*|\s*,\s*|\s+and\s+))*[a-z][a-z\-]+"
    r"\s*,\s*\d{4}[a-z]?"
    r"(?:\s*;\s*[a-z][a-z\-]+(?:\s*&\s*|\s*,\s*|\s+and\s+)*[a-z][a-z\-]+"
    r"\s*,\s*\d{4}[a-z]?)*"
    r")\)", flags=re.I,
)
ETAL_RE = re.compile(r"\b([a-z][a-z\-]+)\s+et\s+al\.?\b", re.I)

def basic_clean(txt: str, lowercase=True, fix_hyphenation=True):
    if lowercase: txt = txt.lower()
    if fix_hyphenation:
        txt = re.sub(r"(?<=\w)-\s+(?=\w)", "", txt)
    txt = URL_RE.sub(" ", txt)
    txt = EMAIL_RE.sub(" ", txt)
    txt = CITATION_RE.sub(" ", txt)
    txt = ETAL_RE.sub(" ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def split_glued_words(token: str):
    parts = re.findall(r"[a-z]+", token)
    return parts if parts else [token]

def process_record(rec, extra_stop=None):
    sw = set(stopwords.words("english")) | set(extra_stop or [])
    text = basic_clean(rec["text"])
    raw_toks = re.findall(r"[a-z][a-z'-]*", text)
    toks = []
    for t in raw_toks:
        toks.extend(split_glued_words(t))
    kept = []
    for w in toks:
        if len(w) <= 2: continue
        if any(ch.isdigit() for ch in w): continue
        if w in sw: continue
        kept.append(WordNetLemmatizer().lemmatize(w))
    rec["tokens"] = kept
    return rec
