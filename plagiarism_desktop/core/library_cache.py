import os
import pickle
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parents[1] / "database" / "vector_store"


def _cache_path(basename, suffix):
    return CACHE_DIR / f"{basename}.{suffix}"


def build_shingle_set(text, k=3):
    words = text.split()
    if len(words) < k:
        return {tuple(words)}
    return {tuple(words[i:i+k]) for i in range(len(words)-k+1)}


def preprocess_and_cache(file_path, basename, text, lang, session):
    try:
        _save_text(basename, text)
        _save_shingles(basename, text)
        _save_tfidf(basename, text, lang, session)
    except Exception as e:
        print(f"[CACHE] Warning: {basename}: {e}")


def _save_text(basename, text):
    p = _cache_path(basename, "text.txt")
    p.write_text(text, encoding="utf-8")


def _save_shingles(basename, text):
    shingle_set = build_shingle_set(text, k=3)
    p = _cache_path(basename, "shingle.pkl")
    with open(p, "wb") as f:
        pickle.dump(shingle_set, f)


def _save_tfidf(basename, text, lang, session):
    from core.preprocessor import preprocess
    tfidf = session.get_tfidf_vectorizer(lang)
    if tfidf is None:
        return
    processed = preprocess(text, lang)
    if not processed:
        return
    vec = tfidf.transform([processed])
    p = _cache_path(basename, "tfidf.pkl")
    with open(p, "wb") as f:
        pickle.dump(vec, f)


def load_text(basename):
    p = _cache_path(basename, "text.txt")
    if p.exists():
        return p.read_text(encoding="utf-8")
    return None


def load_shingle_set(basename):
    p = _cache_path(basename, "shingle.pkl")
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


def load_tfidf_vec(basename):
    p = _cache_path(basename, "tfidf.pkl")
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


def clear_file_cache(basename):
    for suffix in ["text.txt", "shingle.pkl", "tfidf.pkl"]:
        p = _cache_path(basename, suffix)
        if p.exists():
            p.unlink()
