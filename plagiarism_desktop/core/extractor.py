import os
import re
import fitz
import docx
from langdetect import detect, DetectorFactory, LangDetectException

DetectorFactory.seed = 0


def extract_text_from_pdf(filepath):
    text = []
    with fitz.open(filepath) as doc:
        for page in doc:
            page_text = page.get_text()
            if page_text:
                text.append(page_text)
    return "\n".join(text)


def extract_text_from_docx(filepath):
    doc = docx.Document(filepath)
    return "\n".join([p.text for p in doc.paragraphs])


def extract_text_from_txt(filepath):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def extract_text(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(filepath)
    elif ext == ".docx":
        return extract_text_from_docx(filepath)
    elif ext == ".txt":
        return extract_text_from_txt(filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def detect_language(text):
    text_clean = text.strip()[:1000]
    if not text_clean:
        return "en"
    try:
        lang = detect(text_clean)
        if lang == "vi":
            return "vi"
        return "en"
    except LangDetectException:
        return "en"


def chunk_text(text, max_words=256, overlap=50):
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = end - overlap
    return chunks


def get_file_size_mb(filepath):
    return os.path.getsize(filepath) / (1024 * 1024)


def count_words(text):
    return len(text.split())
