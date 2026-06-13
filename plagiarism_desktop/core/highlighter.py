import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# FIX: Ngưỡng nhất quán — paraphrase yêu cầu cả lexical VÀ semantic
# paraphrase_lexical thấp vì PDF hay thiếu space giữa từ tiếng Việt
THRESHOLDS = {
    "copy": 0.95,
    "paraphrase_lexical": 0.35,
    "paraphrase_semantic": 0.88,
}
GAP_TOLERANCE = 2
MIN_WORDS = 8
MAX_MERGED = 50


# === Từ điển loại trừ Common Knowledge (CNTT) ===
_COMMON_KNOWLEDGE_PHRASES_VI = {
    "push", "pop", "peek", "isEmpty", "isempty", "stack", "ngăn xếp",
    "thêm một phần tử mới trên ngăn xếp", "xóa phần tử trên cùng",
    "trả về phần tử đầu tiên", "trả về phần tử trên cùng",
    "kiểm tra xem ngăn xếp có trống không", "kiểm tra xem ngăn xếp có rỗng không",
    "các thao tác cơ bản có thể thực hiện trên một ngăn xếp",
    "enqueue", "dequeue", "queue", "hàng đợi",
    "thêm một phần tử vào cuối hàng đợi", "xóa một phần tử từ đầu hàng đợi",
    "danh sách liên kết", "singly linked list", "doubly linked list",
    "linked list", "danh sách liên kết đơn", "danh sách liên kết kép",
    "danh sách liên kết đôi", "nút trước", "nút tiếp theo",
    "cây nhị phân", "binary tree", "node", "nút gốc", "root",
    "cây tìm kiếm nhị phân", "binary search tree", "bst",
    "hash", "bảng băm", "hash table", "hàm băm", "hash function",
    "sắp xếp nổi bọt", "bubble sort", "sắp xếp chèn", "insertion sort",
    "sắp xếp chọn", "selection sort", "sắp xếp nhanh", "quick sort",
    "sắp xếp trộn", "merge sort", "sắp xếp vun đống", "heap sort",
    "đồ thị", "graph", "bfs", "dfs", "duyệt đồ thị",
    "cấu trúc dữ liệu", "data structure", "giải thuật", "thuật toán",
    "độ phức tạp", "time complexity", "space complexity",
    "big o", "o(n)", "o(log n)", "o(n^2)",
}

_COMMON_KNOWLEDGE_PHRASES_EN = {
    "push", "pop", "peek", "isEmpty", "stack", "queue",
    "enqueue", "dequeue", "linked list", "singly linked list",
    "doubly linked list", "binary tree", "binary search tree",
    "hash table", "hash function", "bubble sort", "insertion sort",
    "selection sort", "quick sort", "merge sort", "heap sort",
    "graph", "bfs", "dfs", "data structure", "algorithm",
    "time complexity", "space complexity", "big o",
}


def _is_common_knowledge(text, language):
    phrases = _COMMON_KNOWLEDGE_PHRASES_VI if language == "vi" else _COMMON_KNOWLEDGE_PHRASES_EN
    lower_text = text.lower()
    return any(phrase in lower_text for phrase in phrases)


def determine_match_type(s1, s3):
    if s1 >= THRESHOLDS["copy"]:
        return "copy"
    if s1 >= THRESHOLDS["paraphrase_lexical"] and s3 >= THRESHOLDS["paraphrase_semantic"]:
        return "paraphrase"
    return None


# ── Tách câu theo ngôn ngữ ─────────────────────────────────────────
def split_sentences(text, language):
    if language == "vi":
        from underthesea import sent_tokenize as vi_sent_tokenize
        return vi_sent_tokenize(text)
    import nltk
    return nltk.sent_tokenize(text)


def split_sentences_with_offsets(text, language):
    sentences = split_sentences(text, language)
    result = []
    pos = 0
    for sent in sentences:
        stripped = sent.strip()
        if not stripped:
            pos += len(sent)
            continue
        start = text.find(stripped, pos)
        if start == -1:
            start = text.find(stripped)
        if start == -1:
            pos += len(sent)
            continue
        end = start + len(stripped)
        result.append({
            "idx": len(result),
            "text": stripped,
            "start": start,
            "end": end,
        })
        pos = end
    return result


# ── Tìm cặp match ──────────────────────────────────────────────────
def find_matching_positions(original_text, suspicious_text, language, session):
    from core.similarity import compute_shingling_similarity

    orig_sents = split_sentences_with_offsets(original_text, language)
    susp_sents = split_sentences_with_offsets(suspicious_text, language)

    if not orig_sents or not susp_sents:
        return []

    orig_long = [s for s in orig_sents if len(s["text"].split()) >= MIN_WORDS]
    susp_long = [s for s in susp_sents if len(s["text"].split()) >= MIN_WORDS]

    if not orig_long or not susp_long:
        return []

    model = session.get_semantic_model(language)
    all_texts = [s["text"] for s in orig_long] + [s["text"] for s in susp_long]
    all_embs = model.encode(all_texts, convert_to_numpy=True)
    orig_embs = all_embs[:len(orig_long)]
    susp_embs = all_embs[len(orig_long):]

    sim_matrix = cosine_similarity(orig_embs, susp_embs)

    matches = []
    for idx_i, o in enumerate(orig_long):
        for idx_j, s in enumerate(susp_long):
            s3 = float(sim_matrix[idx_i][idx_j])
            norm_o = re.sub(r"\s+", " ", o["text"]).strip()
            norm_s = re.sub(r"\s+", " ", s["text"]).strip()
            s1 = compute_shingling_similarity(norm_o, norm_s, k=3)

            # FIX: Copy = lexical rất cao; paraphrase = cả lexical VÀ semantic
            if s1 >= THRESHOLDS["copy"]:
                match_type = "copy"
            elif s1 >= THRESHOLDS["paraphrase_lexical"] and s3 >= THRESHOLDS["paraphrase_semantic"]:
                match_type = "paraphrase"
            else:
                continue
            is_ck = _is_common_knowledge(o["text"], language) or _is_common_knowledge(s["text"], language)

            if is_ck:
                match_type = "common_knowledge"

            matches.append({
                "orig_idx": o["idx"], "susp_idx": s["idx"],
                "orig_start": o["start"], "orig_end": o["end"],
                "susp_start": s["start"], "susp_end": s["end"],
                "orig_text": o["text"], "susp_text": s["text"],
                "score": max(s1, s3), "type": match_type,
                "is_common_knowledge": is_ck,
                "language": language,
            })

    # FIX: Dedup — mỗi câu orig chỉ giữ 1 match tốt nhất
    best_per_orig = {}
    for m in matches:
        key = m["orig_idx"]
        if key not in best_per_orig or m["score"] > best_per_orig[key]["score"]:
            best_per_orig[key] = m
    matches = list(best_per_orig.values())

    merged = _merge_adjacent_matches(matches)
    merged = [m for m in merged
              if len(m["orig_text"].split()) >= MIN_WORDS
              and len(m["susp_text"].split()) >= MIN_WORDS]
    merged.sort(key=lambda m: (m["score"], m["num_sentences"]), reverse=True)
    return merged[:MAX_MERGED]


# ── Gộp câu liền kề ────────────────────────────────────────────────
def _merge_adjacent_matches(matches):
    if not matches:
        return []

    sorted_m = sorted(matches, key=lambda m: (m["orig_idx"], m["susp_idx"]))
    groups = []
    cur = [sorted_m[0]]

    for m in sorted_m[1:]:
        last = cur[-1]
        orig_gap = m["orig_idx"] - last["orig_idx"]
        susp_gap = m["susp_idx"] - last["susp_idx"]
        if (orig_gap >= 0 and susp_gap >= 0 and
                orig_gap <= GAP_TOLERANCE and susp_gap <= GAP_TOLERANCE):
            cur.append(m)
        else:
            groups.append(cur)
            cur = [m]
    groups.append(cur)

    result = []
    for g in groups:
        types = {x["type"] for x in g}
        if "copy" in types:
            merged_type = "copy"
        elif "paraphrase" in types:
            merged_type = "paraphrase"
        else:
            merged_type = "common_knowledge"

        is_ck = any(x.get("is_common_knowledge", False) for x in g)

        orig_indices = sorted({x["orig_idx"] for x in g})
        susp_indices = sorted({x["susp_idx"] for x in g})
        orig_spans = [(x["orig_start"], x["orig_end"]) for x in g]
        susp_spans = [(x["susp_start"], x["susp_end"]) for x in g]

        result.append({
            "orig_text": " ".join(x["orig_text"] for x in g),
            "susp_text": " ".join(x["susp_text"] for x in g),
            "score": max(x["score"] for x in g),
            "type": merged_type,
            "num_sentences": len(g),
            "orig_idx_start": min(orig_indices),
            "orig_idx_end": max(orig_indices),
            "orig_indices": orig_indices,
            "orig_start": g[0]["orig_start"],
            "orig_end": g[-1]["orig_end"],
            "orig_spans": orig_spans,
            "susp_idx_start": min(susp_indices),
            "susp_idx_end": max(susp_indices),
            "susp_indices": susp_indices,
            "susp_start": g[0]["susp_start"],
            "susp_end": g[-1]["susp_end"],
            "susp_spans": susp_spans,
            "is_common_knowledge": is_ck,
            "language": g[0].get("language", "vi"),
        })
    return result


# ── Coverage metric (% số từ bị trùng, như Turnitin) ──────────────
def compute_coverage(original_text, suspicious_text, matches):
    words_a = original_text.split()
    words_b = suspicious_text.split()
    if not words_a or not words_b:
        return 0.0, 0.0

    covered_a = set()
    covered_b = set()

    for m in matches:
        if m.get("is_common_knowledge", False) or m["type"] == "common_knowledge":
            continue
        orig_words = m.get("orig_text", "").split()
        susp_words = m.get("susp_text", "").split()

        # Tìm vị trí của đoạn trùng trong original
        orig_joined = " ".join(orig_words)
        for w_start in range(len(words_a) - len(orig_words) + 1):
            if " ".join(words_a[w_start:w_start + len(orig_words)]) == orig_joined:
                for wi in range(w_start, w_start + len(orig_words)):
                    covered_a.add(wi)
                break

        # Tìm vị trí của đoạn trùng trong suspicious
        susp_joined = " ".join(susp_words)
        for w_start in range(len(words_b) - len(susp_words) + 1):
            if " ".join(words_b[w_start:w_start + len(susp_words)]) == susp_joined:
                for wi in range(w_start, w_start + len(susp_words)):
                    covered_b.add(wi)
                break

    coverage_a = len(covered_a) / len(words_a)
    coverage_b = len(covered_b) / len(words_b)
    return coverage_a, coverage_b


# ── Render HTML highlight ──────────────────────────────────────────
def _remove_overlapping_spans(spans):
    if not spans:
        return []
    sorted_spans = sorted(spans, key=lambda x: (x["start"], -x["end"]))
    result = [sorted_spans[0]]
    for s in sorted_spans[1:]:
        if s["start"] >= result[-1]["end"]:
            result.append(s)
    return result


def render_highlight_html(text, merged_matches, side="orig"):
    import html

    color_map = {
        "copy":             "rgba(255, 68,  68,  0.35)",
        "paraphrase":       "rgba(255, 136, 0,   0.35)",
        "common_knowledge": "rgba(180, 180, 180, 0.35)",
    }
    spans_key = f"{side}_spans"

    spans = []
    for m in merged_matches:
        for start, end in m.get(spans_key, [(m[f"{side}_start"], m[f"{side}_end"])]):
            spans.append({"start": start, "end": end, "type": m["type"], "score": m["score"]})

    spans = _remove_overlapping_spans(spans)
    for span in spans:
        expected = text[span["start"]:span["end"]]
        print(f"[DEBUG] span({span['start']}:{span['end']}) = '{expected[:80]}' type={span['type']} score={span['score']:.2f}")

    result = ""
    prev = 0
    for span in spans:
        if span["start"] < prev:
            continue
        result += html.escape(text[prev:span["start"]])
        color = color_map.get(span["type"], "rgba(255, 255, 0, 0.35)")
        score_pct = int(span["score"] * 100)
        label = span["type"].upper()
        result += (
            f'<span style="background:{color}; border-radius:3px;" '
            f'title="{label} — {score_pct}%">'
            f'{html.escape(text[span["start"]:span["end"]])}'
            f"</span>"
        )
        prev = span["end"]
    result += html.escape(text[prev:])
    return result
