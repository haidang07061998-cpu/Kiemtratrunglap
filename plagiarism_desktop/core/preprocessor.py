import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from underthesea import word_tokenize

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

EN_STOPWORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()


def preprocess_english(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in EN_STOPWORDS and len(t) > 1]
    tokens = [STEMMER.stem(t) for t in tokens]
    return " ".join(tokens)


def preprocess_vietnamese(text):
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text, format="text")
    return tokens


def preprocess(text, language):
    if language == "vi":
        return preprocess_vietnamese(text)
    return preprocess_english(text)
