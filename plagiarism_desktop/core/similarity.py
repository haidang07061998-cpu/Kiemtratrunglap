import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from core.preprocessor import preprocess


def compute_tfidf_similarity(text1, text2, language, session=None):
    processed = [preprocess(text1, language), preprocess(text2, language)]
    if not processed[0] or not processed[1]:
        return 0.0

    # Use pre-trained vectorizer if available
    if session is not None:
        vec = session.get_tfidf_vectorizer(language)
        try:
            m = vec.transform(processed)
            return float(cosine_similarity(m[0:1], m[1:2])[0][0])
        except Exception:
            pass

    # Fallback: fit on-the-fly
    vec = TfidfVectorizer()
    try:
        m = vec.fit_transform(processed)
        return float(cosine_similarity(m[0:1], m[1:2])[0][0])
    except ValueError:
        return 0.0


def compute_shingling_similarity(text1, text2, k=3):
    def get_shingles(text, k):
        words = text.split()
        if len(words) < k:
            return {tuple(words)}
        return {tuple(words[i:i+k]) for i in range(len(words)-k+1)}

    s1 = get_shingles(text1, k)
    s2 = get_shingles(text2, k)
    if not s1 or not s2:
        return 0.0
    intersection = s1 & s2
    union = s1 | s2
    return len(intersection) / len(union)


def compute_semantic_similarity(text1, text2, language, session):
    if not text1.strip() or not text2.strip():
        return 0.0
    try:
        # Ưu tiên cross-encoder (chính xác hơn)
        if language != "vi" and session.cross_encoder_en is not None:
            return float(session.cross_encoder_en.predict([(text1[:2000], text2[:2000])])[0])
        # Fallback: bi-encoder + cosine
        model = session.get_semantic_model(language)
        emb1 = model.encode([text1], convert_to_numpy=True)
        emb2 = model.encode([text2], convert_to_numpy=True)
        sim = cosine_similarity(emb1, emb2)[0][0]
        return float(sim)
    except Exception:
        return 0.0


def compute_ensemble_score(text1, text2, language, session):
    s1 = compute_tfidf_similarity(text1, text2, language, session)
    s2 = compute_shingling_similarity(text1, text2, k=3)
    s3 = compute_semantic_similarity(text1, text2, language, session)

    # Trọng số động — Giải pháp 2:
    # Nếu BERT > 90% hoặc TF-IDF > 90% → paraphrase có chủ đích
    #   ưu tiên: Ngữ nghĩa × 0.5 + Từ khóa × 0.3 + Nguyên văn × 0.2
    # Ngược lại → mặc định: Nguyên văn × 0.5 + Từ khóa × 0.3 + Ngữ nghĩa × 0.2
    if s3 > 0.9 or s1 > 0.9:
        final = s3 * 0.5 + s1 * 0.3 + s2 * 0.2
    else:
        final = s2 * 0.5 + s1 * 0.3 + s3 * 0.2

    return final, s1, s2, s3


def classify_plagiarism(score, s1=None, s2=None, s3=None):
    # Cảnh báo: Shingling thấp + Semantic cao = cùng chủ đề
    if s2 is not None and s2 < 0.1 and s3 is not None and s3 > 0.85:
        return "⚠ Trùng chủ đề (Cùng lĩnh vực, không đạo văn)"

    # Ngưỡng mới theo đề xuất:
    # >= 95%: Copy-paste hoàn toàn
    # 50-95%: Paraphrase
    # < 50%:  Bỏ qua / Low similarity
    if score >= 0.95:
        return "🚨 Sao chép hoàn toàn (Copy-paste)"
    elif score >= 0.50:
        return "🔶 Có chỉnh sửa (Paraphrase)"
    else:
        return "✅ Ít trùng lặp (Không đáng kể)"
