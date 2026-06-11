import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# === Giải pháp 3: Từ điển loại trừ Common Knowledge (CNTT) ===
_COMMON_KNOWLEDGE_PHRASES_VI = {
    # Stack
    "push", "pop", "peek", "isEmpty", "isempty", "stack", "ngăn xếp",
    "thêm một phần tử mới trên ngăn xếp", "xóa phần tử trên cùng",
    "trả về phần tử đầu tiên", "trả về phần tử trên cùng",
    "kiểm tra xem ngăn xếp có trống không", "kiểm tra xem ngăn xếp có rỗng không",
    "các thao tác cơ bản có thể thực hiện trên một ngăn xếp",
    # Queue
    "enqueue", "dequeue", "queue", "hàng đợi",
    "thêm một phần tử vào cuối hàng đợi", "xóa một phần tử từ đầu hàng đợi",
    # Linked List
    "danh sách liên kết", "singly linked list", "doubly linked list",
    "linked list", "danh sách liên kết đơn", "danh sách liên kết kép",
    "danh sách liên kết đôi", "nút trước", "nút tiếp theo",
    # Tree
    "cây nhị phân", "binary tree", "node", "nút gốc", "root",
    "cây tìm kiếm nhị phân", "binary search tree", "bst",
    # Hash
    "hash", "bảng băm", "hash table", "hàm băm", "hash function",
    # Sorting
    "sắp xếp nổi bọt", "bubble sort", "sắp xếp chèn", "insertion sort",
    "sắp xếp chọn", "selection sort", "sắp xếp nhanh", "quick sort",
    "sắp xếp trộn", "merge sort", "sắp xếp vun đống", "heap sort",
    # Graph
    "đồ thị", "graph", "bfs", "dfs", "duyệt đồ thị",
    # General DS
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


def find_matching_positions(original_text, suspicious_text, language, session):
    from core.preprocessor import preprocess
    from core.similarity import compute_shingling_similarity

    # Split by both punctuation and paragraph breaks
    orig_sentences = re.split(r'(?<=[.!?])\s+|\n\s*\n', original_text)
    susp_sentences = re.split(r'(?<=[.!?])\s+|\n\s*\n', suspicious_text)

    orig_sentences = [s.strip() for s in orig_sentences if s.strip()]
    susp_sentences = [s.strip() for s in susp_sentences if s.strip()]

    if not orig_sentences or not susp_sentences:
        return []

    # Giải pháp 2: Bộ lọc độ dài tối thiểu (≥8 từ / câu)
    def is_long_enough(s):
        return len(s.split()) >= 8
    orig_long = [(i, s) for i, s in enumerate(orig_sentences) if is_long_enough(s)]
    susp_long = [(j, s) for j, s in enumerate(susp_sentences) if is_long_enough(s)]

    if not orig_long or not susp_long:
        return []

    model = session.get_semantic_model(language)
    all_texts = [s for _, s in orig_long] + [s for _, s in susp_long]
    all_embs = model.encode(all_texts, convert_to_numpy=True)
    orig_embs = all_embs[:len(orig_long)]
    susp_embs = all_embs[len(orig_long):]

    sim_matrix = cosine_similarity(orig_embs, susp_embs)

    # Giải pháp 1: Ngưỡng mới
    #   < 50% → bỏ qua
    #   50-95% → paraphrase
    #   >= 95% → copy
    MIN_SIM = 0.5
    COPY_THRESHOLD = 0.95

    matches = []
    for idx_i, (orig_idx, osent) in enumerate(orig_long):
        for idx_j, (susp_idx, ssent) in enumerate(susp_long):
            s3 = float(sim_matrix[idx_i][idx_j])
            s1 = compute_shingling_similarity(osent, ssent, k=3)

            if s1 < MIN_SIM:
                continue

            label = "copy" if s1 >= COPY_THRESHOLD else "paraphrase"

            is_ck = _is_common_knowledge(osent, language) or _is_common_knowledge(ssent, language)

            matches.append({
                "orig_idx": orig_idx, "susp_idx": susp_idx,
                "orig_text": osent, "susp_text": ssent,
                "score": max(s1, s3), "type": label,
                "is_common_knowledge": is_ck,
            })

    matches.sort(key=lambda x: x["score"], reverse=True)
    matches = matches[:50]

    merged = _merge_adjacent_matches(matches)
    # Giải pháp 2: Lọc block < 6 từ
    merged = [m for m in merged if len(m["orig_text"].split()) >= 6 and len(m["susp_text"].split()) >= 6]
    merged.sort(key=lambda m: (m["score"], m["num_sentences"]), reverse=True)
    return merged[:15]


def _merge_adjacent_matches(matches):
    if not matches:
        return []

    sorted_m = sorted(matches, key=lambda m: (m["orig_idx"], m["susp_idx"]))
    groups = []
    cur = [sorted_m[0]]

    for m in sorted_m[1:]:
        last = cur[-1]
        if m["orig_idx"] - last["orig_idx"] == 1 and m["susp_idx"] - last["susp_idx"] == 1:
            cur.append(m)
        else:
            groups.append(cur)
            cur = [m]
    groups.append(cur)

    result = []
    for g in groups:
        is_ck = any(x.get("is_common_knowledge", False) for x in g)
        result.append({
            "orig_text": " ".join(x["orig_text"] for x in g),
            "susp_text": " ".join(x["susp_text"] for x in g),
            "score": max(x["score"] for x in g),
            "type": "copy" if any(x["type"] == "copy" for x in g) else "paraphrase",
            "num_sentences": len(g),
            "orig_idx_start": g[0]["orig_idx"],
            "orig_idx_end": g[-1]["orig_idx"],
            "susp_idx_start": g[0]["susp_idx"],
            "susp_idx_end": g[-1]["susp_idx"],
            "is_common_knowledge": is_ck,
        })
    return result


def determine_match_type(score, s1, s2, s3):
    if score >= 0.95:
        return "copy"
    elif score >= 0.50:
        return "paraphrase"
    else:
        return "similar"
