"""
Sinh pan_training_pairs.csv từ PAN-PC-11 corpus
Output: 2000 positive pairs + 1000 negative pairs
"""

import os, re, random, xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

PAN_DIR = r"F:\Kiemtratrunglap\pan-plagiarism-corpus-2011"
SUSP_DIR = os.path.join(PAN_DIR, "external-detection-corpus", "suspicious-document")
SRC_PARTS = os.path.join(PAN_DIR, "external-detection-corpus", "source-document")

OUTPUT = r"F:\Kiemtratrunglap\plagiarism_desktop\pan_training_pairs.csv"
SEED = 42
MAX_POS = 2000
MAX_NEG = 1000
random.seed(SEED)

def extract_text(filepath, offset, length):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return text[offset:offset + length]

def parse_ann(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    anns = []
    for feat in root.findall("feature"):
        if feat.get("name") == "plagiarism":
            anns.append({
                "this_offset": int(feat.get("this_offset", 0)),
                "this_length": int(feat.get("this_length", 0)),
                "source_reference": feat.get("source_reference", ""),
                "source_offset": int(feat.get("source_offset", 0)),
                "source_length": int(feat.get("source_length", 0)),
            })
    return anns

def find_source(source_ref):
    for sp in sorted(os.listdir(SRC_PARTS)):
        p = os.path.join(SRC_PARTS, sp, source_ref)
        if os.path.exists(p):
            return p
    return None

# Thu thập positive pairs
positive_pairs = []
parts = sorted([p for p in os.listdir(SUSP_DIR) if p.startswith("part")])
for part in parts:
    part_path = os.path.join(SUSP_DIR, part)
    if not os.path.isdir(part_path):
        continue
    for xmlf in sorted(os.listdir(part_path)):
        if not xmlf.endswith(".xml"):
            continue
        xml_path = os.path.join(part_path, xmlf)
        anns = parse_ann(xml_path)
        if not anns:
            continue
        txt_path = xml_path.replace(".xml", ".txt")
        if not os.path.exists(txt_path):
            continue
        for ann in anns:
            src_path = find_source(ann["source_reference"])
            if not src_path:
                continue
            susp_text = extract_text(txt_path, ann["this_offset"], ann["this_length"])
            src_text = extract_text(src_path, ann["source_offset"], ann["source_length"])
            if len(susp_text.strip()) < 20 or len(src_text.strip()) < 20:
                continue
            positive_pairs.append({
                "text_a": susp_text[:2000],
                "text_b": src_text[:2000],
                "label": 1
            })
            if len(positive_pairs) >= MAX_POS:
                break
        if len(positive_pairs) >= MAX_POS:
            break
    if len(positive_pairs) >= MAX_POS:
        break

print(f"Positive pairs: {len(positive_pairs)}")

# Hard negative mining (chọn cặp khác chủ đề gần)
all_texts = [p["text_a"] for p in positive_pairs[:1500]]
print("Hard negative mining...")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

proc_texts = [re.sub(r"[^a-z\s]", "", t.lower()) for t in all_texts if t.strip()]
proc_texts = list(set(proc_texts))
vec = TfidfVectorizer()
tfidf_matrix = vec.fit_transform(proc_texts)
sim_matrix = cos_sim(tfidf_matrix)
sim_matrix = sim_matrix - np.eye(sim_matrix.shape[0]) * 2

hard_neg = []
used = set()
for i in range(min(1500, len(proc_texts))):
    if i in used:
        continue
    candidates = [(j, sim_matrix[i][j]) for j in range(len(proc_texts))
                  if j != i and j not in used and sim_matrix[i][j] > 0.05]
    if candidates:
        candidates.sort(key=lambda x: -x[1])
        best_j = candidates[0][0]
        hard_neg.append({"text_a": proc_texts[i], "text_b": proc_texts[best_j], "label": 0})
        used.add(i); used.add(best_j)

neg_pairs = hard_neg[:MAX_NEG]
while len(neg_pairs) < MAX_NEG:
    j = random.randint(0, len(positive_pairs) - 1)
    k = random.randint(0, len(positive_pairs) - 1)
    if j != k:
        neg_pairs.append({"text_a": positive_pairs[j]["text_a"], "text_b": positive_pairs[k]["text_b"], "label": 0})

print(f"Negative pairs: {len(neg_pairs)}")

# Gộp và lưu
subset = positive_pairs[:MAX_POS] + neg_pairs
random.shuffle(subset)
df = pd.DataFrame(subset)
df.to_csv(OUTPUT, index=False)
print(f"Saved: {OUTPUT} ({len(df)} rows)")
print(f"  Pos: {sum(df['label'] == 1)}, Neg: {sum(df['label'] == 0)}")
