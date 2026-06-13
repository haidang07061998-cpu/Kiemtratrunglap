import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from core.preprocessor import preprocess


def compute_tfidf_similarity(text1, text2, language, session=None):
    processed = [preprocess(text1, language), preprocess(text2, language)]
    if not processed[0] or not processed[1]:
        return 0.0

    def _score(vec):
        import logging
        try:
            m = vec.transform(processed)
            return max(0.0, min(1.0, float(cosine_similarity(m[0:1], m[1:2])[0][0])))
        except Exception as e:
            logging.getLogger(__name__).warning(f"TF-IDF failed: {e}")
            return None

    if session is not None:
        score = _score(session.get_tfidf_vectorizer(language))
        if score is not None:
            return score

    score = _score(TfidfVectorizer())
    return score if score is not None else 0.0


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
        return max(0.0, min(1.0, float(sim)))
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"compute_semantic_similarity failed: {e}")
        import traceback; traceback.print_exc()
        return 0.0


def compute_ensemble_score(text1, text2, language, session):
    s1 = compute_tfidf_similarity(text1, text2, language, session)
    s2 = compute_shingling_similarity(text1, text2, k=3)
    s3 = compute_semantic_similarity(text1, text2, language, session)

    # Fallback: trọng số động
    if s3 > 0.9 or s1 > 0.9:
        fallback = s3 * 0.5 + s1 * 0.3 + s2 * 0.2
    else:
        fallback = s2 * 0.5 + s1 * 0.3 + s3 * 0.2

    # Chỉ dùng ensemble model nếu kết quả không quá chênh lệch với fallback
    ensemble = session.get_ensemble_model(language)
    if ensemble is not None and "classifier" in ensemble:
        try:
            len_ratio = _safe_len_ratio(text1, text2)
            features = np.array([[s1, s2, s3, len_ratio]])
            clf = ensemble["classifier"]
            proba = float(clf.predict_proba(features)[0][1])
            if abs(proba - fallback) < 0.15:
                final = (proba + fallback) / 2
            else:
                final = fallback
            return final, s1, s2, s3
        except Exception:
            pass

    return fallback, s1, s2, s3


def _safe_len_ratio(text1, text2):
    len1, len2 = len(text1.split()), len(text2.split())
    if max(len1, len2) == 0:
        return 0.0
    return min(len1, len2) / max(len1, len2)


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
