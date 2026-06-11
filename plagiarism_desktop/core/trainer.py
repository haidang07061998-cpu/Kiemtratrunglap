import os
import csv
import xml.etree.ElementTree as ET
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, accuracy_score
import pickle

from core.extractor import extract_text
from core.preprocessor import preprocess
from core.similarity import (
    compute_tfidf_similarity,
    compute_shingling_similarity,
    compute_semantic_similarity,
)
from gui.session import AppSession

PAN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                       "pan-plagiarism-corpus-2011")
VISP_TRAIN = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                           "visp_train.csv")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          "models")
TRAINED_PATH_EN = os.path.join(MODELS_DIR, "ensemble_model_pan.pkl")
TRAINED_PATH_VI = os.path.join(MODELS_DIR, "ensemble_model_vi.pkl")


def extract_pan_text_by_offset(filepath, offset, length):
    text = extract_text(filepath)
    lines = text.splitlines(keepends=True)
    char_count = 0
    for line in lines:
        line_len = len(line)
        if char_count + line_len > offset:
            start_in_line = offset - char_count
            end_in_line = start_in_line + length
            return line[start_in_line:end_in_line]
        char_count += line_len
    return ""


def parse_pan_annotation(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ref = root.get("reference", "")
    annotations = []
    for feature in root.findall("feature"):
        ftype = feature.get("name", "")
        if ftype == "plagiarism":
            ann = {
                "type": feature.get("type", ""),
                "obfuscation": feature.get("obfuscation", ""),
                "this_offset": int(feature.get("this_offset", 0)),
                "this_length": int(feature.get("this_length", 0)),
                "source_reference": feature.get("source_reference", ""),
                "source_offset": int(feature.get("source_offset", 0)),
                "source_length": int(feature.get("source_length", 0)),
            }
            annotations.append(ann)
    return ref, annotations


def collect_pan_pairs(max_samples=2000, progress_callback=None):
    pairs = []
    susp_dir = os.path.join(PAN_DIR, "external-detection-corpus",
                            "suspicious-document")
    source_dir = os.path.join(PAN_DIR, "external-detection-corpus",
                              "source-document")

    count = 0
    parts = sorted([p for p in os.listdir(susp_dir) if p.startswith("part")])

    for pi, part in enumerate(parts):
        part_path = os.path.join(susp_dir, part)
        if not os.path.isdir(part_path):
            continue
        xml_files = sorted([f for f in os.listdir(part_path)
                            if f.endswith(".xml") and f.startswith("suspicious")])
        total_xml = len(xml_files)
        for xi, xmlf in enumerate(xml_files):
            if progress_callback and xi % 100 == 0:
                progress_callback(pi, part, xi, total_xml, len(pairs))
            xml_path = os.path.join(part_path, xmlf)
            ref, annotations = parse_pan_annotation(xml_path)
            if not annotations:
                continue
            susp_txt_path = os.path.join(part_path,
                                         ref.replace(".xml", ".txt"))
            if not os.path.exists(susp_txt_path):
                continue
            for ann in annotations:
                src_ref = ann["source_reference"]
                src_part = None
                for p in os.listdir(source_dir):
                    sp = os.path.join(source_dir, p, src_ref)
                    if os.path.exists(sp):
                        src_part = p
                        break
                if not src_part:
                    continue
                src_path = os.path.join(source_dir, src_part, src_ref)
                if not os.path.exists(src_path):
                    continue
                susp_text = extract_pan_text_by_offset(
                    susp_txt_path, ann["this_offset"], ann["this_length"]
                )
                src_text = extract_pan_text_by_offset(
                    src_path, ann["source_offset"], ann["source_length"]
                )
                if len(susp_text.strip()) < 20 or len(src_text.strip()) < 20:
                    continue
                pairs.append({
                    "text_a": susp_text,
                    "text_b": src_text,
                    "label": 1,
                    "obfuscation": ann.get("obfuscation", "none"),
                })
                count += 1
                if max_samples and count >= max_samples:
                    return pairs
    return pairs


def collect_visp_pairs(max_samples=50000):
    pairs = []
    if not os.path.exists(VISP_TRAIN):
        return pairs

    with open(VISP_TRAIN, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get("paraphrase_id", "")
            if pid.startswith("rpara-"):
                continue
            if pid.startswith("para-"):
                pairs.append({
                    "text_a": row.get("original_text", ""),
                    "text_b": row.get("paraphrase_text", ""),
                    "label": 1,
                    "source": row.get("source", ""),
                })
            if max_samples and len(pairs) >= max_samples:
                break

    neg_pairs = []
    pos_samples = len(pairs)
    texts = [(p["text_a"], p["text_b"]) for p in pairs]
    random.shuffle(texts)
    for i in range(min(len(texts), pos_samples)):
        j = random.randint(0, len(texts) - 1)
        if j != i:
            neg_pairs.append({
                "text_a": texts[i][0],
                "text_b": texts[(j + 1) % len(texts)][1],
                "label": 0,
                "source": "negative",
            })
    pairs.extend(neg_pairs)
    random.shuffle(pairs)
    return pairs


def extract_features(text_a, text_b, language, session):
    s1 = compute_tfidf_similarity(text_a, text_b, language, session)
    s2 = compute_shingling_similarity(text_a, text_b, k=3)
    s3 = compute_semantic_similarity(text_a, text_b, language, session)
    len_ratio = min(len(text_a.split()), len(text_b.split())) / \
        max(len(text_a.split()), len(text_b.split())) if max(
            len(text_a.split()), len(text_b.split())) > 0 else 0
    return np.array([s1, s2, s3, len_ratio])


def generate_negative_pan_pairs(pairs, ratio=0.5):
    num_neg = int(len(pairs) * ratio)
    negs = []
    for i in range(num_neg):
        p1 = random.choice(pairs)
        p2 = random.choice(pairs)
        if p1["text_a"] != p2["text_b"]:
            negs.append({
                "text_a": p1["text_a"],
                "text_b": p2["text_b"],
                "label": 0,
                "obfuscation": "negative",
            })
    return negs


def train_ensemble(session, progress_callback=None, language="en"):
    if language == "vi":
        return _train_ensemble_vi(session, progress_callback)
    return _train_ensemble_en(session, progress_callback)


def _train_ensemble_en(session, progress_callback=None):
    if progress_callback:
        progress_callback(5, "Đang thu thập dữ liệu PAN...")
    pan_pos = collect_pan_pairs(max_samples=1500, progress_callback=None)
    if progress_callback:
        progress_callback(20, f"Thu thập xong {len(pan_pos)} cặp PAN")

    if progress_callback:
        progress_callback(25, "Đang sinh negative samples...")
    pan_neg = generate_negative_pan_pairs(pan_pos, ratio=0.5)

    all_pairs = pan_pos + pan_neg

    if progress_callback:
        progress_callback(30, f"Đang trích xuất features cho {len(all_pairs)} cặp...")

    X, y = [], []
    total = len(all_pairs)
    for i, pair in enumerate(all_pairs):
        lang = "en"
        feats = extract_features(pair["text_a"], pair["text_b"], lang, session)
        X.append(feats)
        y.append(pair["label"])

        if progress_callback and i % 50 == 0:
            pct = 30 + int((i + 1) / total * 50)
            progress_callback(pct, f"Đang xử lý {i+1}/{total}...")

    X = np.array(X)
    y = np.array(y)

    if progress_callback:
        progress_callback(80, "Đang huấn luyện Logistic Regression...")

    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(clf, X, y, cv=kfold, scoring=['accuracy', 'f1'])

    acc = cv_results['test_accuracy'].mean()
    acc_std = cv_results['test_accuracy'].std()
    f1 = cv_results['test_f1'].mean()
    f1_std = cv_results['test_f1'].std()

    clf.fit(X, y)
    y_pred = clf.predict(X)
    report = classification_report(y, y_pred)

    if progress_callback:
        progress_callback(90, "Đang lưu model...")

    model_data = {
        "classifier": clf,
        "feature_names": ["tfidf", "shingling", "semantic", "len_ratio"],
        "accuracy": acc,
        "cv_scores": cv_results['test_accuracy'].tolist(),
        "cv_std": acc_std,
        "f1": f1,
        "f1_std": f1_std,
        "report": report,
    }
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(TRAINED_PATH_EN, "wb") as f:
        pickle.dump(model_data, f)

    if progress_callback:
        progress_callback(100, f"Hoàn thành! Accuracy: {acc*100:.2f}% (±{acc_std*100:.2f}%) | F1: {f1*100:.2f}% (±{f1_std*100:.2f}%)")

    return model_data


def _train_ensemble_vi(session, progress_callback=None):
    if progress_callback:
        progress_callback(5, "Đang thu thập dữ liệu VISP...")
    pairs = collect_visp_pairs(max_samples=50000)
    if progress_callback:
        progress_callback(30, f"Thu thập xong {len(pairs)} cặp VISP")

    if progress_callback:
        progress_callback(35, f"Đang trích xuất features...")

    X, y = [], []
    total = len(pairs)
    for i, pair in enumerate(pairs):
        feats = extract_features(pair["text_a"], pair["text_b"], "vi", session)
        X.append(feats)
        y.append(pair["label"])

        if progress_callback and i % 100 == 0:
            pct = 35 + int((i + 1) / total * 45)
            progress_callback(pct, f"Đang xử lý {i+1}/{total}...")

    X = np.array(X)
    y = np.array(y)

    if progress_callback:
        progress_callback(80, "Đang huấn luyện...")

    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(clf, X, y, cv=kfold, scoring=['accuracy', 'f1'])

    acc = cv_results['test_accuracy'].mean()
    acc_std = cv_results['test_accuracy'].std()
    f1 = cv_results['test_f1'].mean()
    f1_std = cv_results['test_f1'].std()

    clf.fit(X, y)
    y_pred = clf.predict(X)
    report = classification_report(y, y_pred)

    if progress_callback:
        progress_callback(90, "Đang lưu model...")

    model_data = {
        "classifier": clf,
        "feature_names": ["tfidf", "shingling", "semantic", "len_ratio"],
        "accuracy": acc,
        "cv_scores": cv_results['test_accuracy'].tolist(),
        "cv_std": acc_std,
        "f1": f1,
        "f1_std": f1_std,
        "report": report,
    }
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(TRAINED_PATH_VI, "wb") as f:
        pickle.dump(model_data, f)

    if progress_callback:
        progress_callback(100, f"Hoàn thành! Accuracy: {acc*100:.2f}% (±{acc_std*100:.2f}%) | F1: {f1*100:.2f}% (±{f1_std*100:.2f}%)")

    return model_data


def load_trained_model(language="en"):
    path = TRAINED_PATH_EN if language == "en" else TRAINED_PATH_VI
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def predict_ensemble(features, language="en"):
    model_data = load_trained_model(language)
    if model_data:
        clf = model_data["classifier"]
        proba = clf.predict_proba(features.reshape(1, -1))[0][1]
        return float(proba)
    return None
