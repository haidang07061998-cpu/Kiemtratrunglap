"""
====================================================================
  KAGGLE NOTEBOOK — ViSP FULL ENSEMBLE PIPELINE (FIXED)
  TF-IDF + Shingling + PhoBERT + Logistic Regression
  Kèm sinh Confusion Matrix và biểu đồ cho báo cáo

  SỬA SO VỚI BẢN CŨ:
  1. Negative sampling dùng index-based (không ghép cặp dương)
  2. Thêm TF-IDF + Shingling features + Logistic Regression ensemble
  3. Sinh Confusion Matrix (Hình 4.2) + biểu đồ metrics
====================================================================
"""

# === 1. Install ===
!pip install -q sentence-transformers scikit-learn pandas numpy torch tqdm underthesea matplotlib seaborn

# === 2. Imports ===
import os, pickle, csv, random, shutil, warnings
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    roc_curve, auc, precision_recall_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

OUTPUT_DIR = "/kaggle/working/visp_model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 3. LOAD VISP ===
print("\n" + "="*60)
print("  [1/7] LOAD VISP DATA")
print("="*60)

import glob
train_path = test_path = None
for f in glob.glob("/kaggle/input/**/*.csv", recursive=True):
    if "visp_train" in f: train_path = f
    if "visp_test" in f: test_path = f

if not train_path or not test_path:
    raise FileNotFoundError("Upload visp_train.csv and visp_test.csv to Kaggle dataset!")

print(f"  Train: {train_path}")
print(f"  Test:  {test_path}")

def load_pos(path, max_pairs=50000):
    pos = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for row in csv.DictReader(f):
            pid = row.get("paraphrase_id", "")
            if pid.startswith("para-"):
                pos.append({
                    "text_a": row.get("original_text", ""),
                    "text_b": row.get("paraphrase_text", ""),
                    "label": 1,
                    "source": row.get("source", ""),
                })
                if len(pos) >= max_pairs:
                    break
    return pos

# FIX: negative sampling bằng index (không ghép cặp dương)
def gen_neg_fixed(pos_list, seed=42):
    rng = random.Random(seed)
    n = len(pos_list)
    indices_a = list(range(n))
    indices_b = list(range(n))
    rng.shuffle(indices_a)
    rng.shuffle(indices_b)
    neg = []
    for i in range(n):
        ia, ib = indices_a[i], indices_b[i]
        if ia == ib:
            ib = (ib + 1) % n
        a = pos_list[ia]["text_a"]
        b = pos_list[ib]["text_b"]
        if a == b:
            continue
        neg.append({"text_a": a, "text_b": b, "label": 0, "source": "negative"})
    return neg

train_pos = load_pos(train_path, max_pairs=50000)
test_pos = load_pos(test_path, max_pairs=10000)

train_neg = gen_neg_fixed(train_pos, seed=42)
test_neg = gen_neg_fixed(test_pos, seed=43)

train_pairs = train_pos + train_neg
random.shuffle(train_pairs)
test_pairs = test_pos + test_neg
random.shuffle(test_pairs)

print(f"  Train: {len(train_pairs)} ({len(train_pos)} pos + {len(train_neg)} neg)")
print(f"  Test:  {len(test_pairs)} ({len(test_pos)} pos + {len(test_neg)} neg)")

# === 4. PHOBERT FINE-TUNE ===
print("\n" + "="*60)
print("  [2/7] PHOBERT FINE-TUNE")
print("="*60)

phobert = SentenceTransformer("vinai/phobert-base", device=DEVICE)

examples = [InputExample(texts=[p["text_a"][:2000], p["text_b"][:2000]], label=float(p["label"]))
            for p in train_pairs[:10000]]

phobert.fit(
    train_objectives=[(DataLoader(examples, shuffle=True, batch_size=16),
                       losses.CosineSimilarityLoss(model=phobert))],
    epochs=3, warmup_steps=50,
    output_path="/kaggle/working/phobert_finetuned",
    show_progress_bar=True, save_best_model=True,
)
print("  PhoBERT fine-tuned!")

# === 5. TF-IDF VECTORIZER ===
print("\n" + "="*60)
print("  [3/7] TF-IDF VECTORIZER")
print("="*60)

def preprocess_vi(text):
    try:
        from underthesea import word_tokenize
        tokens = word_tokenize(text, format="text")
    except:
        tokens = text
    return tokens.lower()

train_texts = []
for p in train_pairs:
    train_texts.append(preprocess_vi(p["text_a"]))
    train_texts.append(preprocess_vi(p["text_b"]))

tfidf_vec = TfidfVectorizer(max_features=5000, sublinear_tf=True)
tfidf_vec.fit(train_texts)
print(f"  TF-IDF vocab: {len(tfidf_vec.vocabulary_)}")

with open(os.path.join(OUTPUT_DIR, "tfidf_vectorizer_vi.pkl"), "wb") as f:
    pickle.dump(tfidf_vec, f)

# === 6. FEATURE EXTRACTION ===
print("\n" + "="*60)
print("  [4/7] EXTRACT FEATURES")
print("="*60)

def extract_vi_features(text_a, text_b):
    # TF-IDF
    pa, pb = preprocess_vi(text_a), preprocess_vi(text_b)
    s1 = 0.0
    if pa and pb:
        try:
            m = tfidf_vec.transform([pa, pb])
            s1 = float(cos_sim(m[0:1], m[1:2])[0][0])
        except:
            pass
    # Shingling
    def sh(t, k=3):
        w = t.split()
        if len(w) < k:
            return {tuple(w)}
        return {tuple(w[i:i+k]) for i in range(len(w)-k+1)}
    sh1, sh2 = sh(text_a), sh(text_b)
    s2 = len(sh1 & sh2) / len(sh1 | sh2) if sh1 and sh2 else 0.0
    # Semantic (PhoBERT)
    emb = phobert.encode([text_a[:2000], text_b[:2000]], convert_to_numpy=True)
    s3 = float(cos_sim(emb[0:1], emb[1:2])[0][0])
    # Length ratio
    la, lb = len(text_a.split()), len(text_b.split())
    s4 = min(la, lb) / max(la, lb) if max(la, lb) > 0 else 0
    return [s1, s2, s3, s4]

X_train = np.array([extract_vi_features(p["text_a"], p["text_b"]) for p in tqdm(train_pairs, desc="Train features")])
y_train = np.array([p["label"] for p in train_pairs])
X_test = np.array([extract_vi_features(p["text_a"], p["text_b"]) for p in tqdm(test_pairs, desc="Test features")])
y_test = np.array([p["label"] for p in test_pairs])
print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

# === 7. TRAIN LOGISTIC REGRESSION ===
print("\n" + "="*60)
print("  [5/7] TRAIN LOGISTIC REGRESSION")
print("="*60)

grid = GridSearchCV(
    LogisticRegression(max_iter=2000),
    {"C": [0.01, 0.1, 1, 10], "class_weight": ["balanced", None]},
    cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
    scoring="f1", n_jobs=-1, verbose=0
)
grid.fit(X_train, y_train)
clf = grid.best_estimator_
print(f"  Best params: {grid.best_params_}")

# === 8. EVALUATE ===
print("\n" + "="*60)
print("  [6/7] EVALUATE ON TEST SET")
print("="*60)

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")

print(f"  Accuracy:  {acc*100:.2f}%")
print(f"  Precision: {prec*100:.2f}%")
print(f"  Recall:    {rec*100:.2f}%")
print(f"  F1:        {f1*100:.2f}%")
print(f"\n{classification_report(y_test, y_pred, target_names=['Non-plagiarized', 'Plagiarized'])}")

# === 9. GENERATE IMAGES FOR REPORT ===
print("\n" + "="*60)
print("  [7/7] GENERATE REPORT FIGURES")
print("="*60)

# Hình 4.2: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Không đạo văn", "Đạo văn"])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap="Blues", values_format="d")
ax.set_title("Hình 4.2: Confusion Matrix — Mô hình ViSP", fontsize=13, pad=15)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "hinh_4_2_confusion_matrix_visp.png"), dpi=200)
plt.close()
print("  Saved: hinh_4_2_confusion_matrix_visp.png")

# Biểu đồ metrics (F1, Precision, Recall, Accuracy)
fig, ax = plt.subplots(figsize=(8, 5))
metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
metrics_values = [acc * 100, prec * 100, rec * 100, f1 * 100]
colors_metrics = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"]
bars = ax.bar(metrics_names, metrics_values, color=colors_metrics, edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, metrics_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_ylim(0, 105)
ax.set_ylabel("Tỷ lệ (%)", fontsize=12)
ax.set_title("Kết quả đánh giá mô hình ViSP (Tiếng Việt)", fontsize=13, pad=15)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "hinh_4_2_metrics_visp.png"), dpi=200)
plt.close()
print("  Saved: hinh_4_2_metrics_visp.png")

# === HÌNH 4.X: CÁC BIỂU ĐỒ BỔ SUNG ===

# 1. ROC Curve + AUC
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color="#3498db", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
ax.fill_between(fpr, tpr, alpha=0.15, color="#3498db")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curve — ViSP", fontsize=13, pad=15)
ax.legend(loc="lower right", fontsize=11)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "hinh_4_roc_curve_visp.png"), dpi=200)
plt.close()
print("  Saved: hinh_4_roc_curve_visp.png")

# 2. Precision-Recall Curve
precisions, recalls, _ = precision_recall_curve(y_test, y_proba)
pr_auc = auc(recalls, precisions)
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(recalls, precisions, color="#e74c3c", lw=2, label=f"PR curve (AUC = {pr_auc:.4f})")
ax.fill_between(recalls, precisions, alpha=0.15, color="#e74c3c")
ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title("Precision-Recall Curve — ViSP", fontsize=13, pad=15)
ax.legend(loc="lower left", fontsize=11)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "hinh_4_pr_curve_visp.png"), dpi=200)
plt.close()
print("  Saved: hinh_4_pr_curve_visp.png")

# 3. Histogram điểm số theo lớp
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
feature_labels = ["TF-IDF Similarity", "Shingling Similarity", "Semantic Similarity"]
colors_hist = {0: "#e74c3c", 1: "#3498db"}
for fi in range(3):
    ax = axes[fi]
    for label in [0, 1]:
        vals = X_test[y_test == label, fi]
        ax.hist(vals, bins=30, alpha=0.6, color=colors_hist[label],
                label=f"{'Plagiarized' if label==1 else 'Non-plagiarized'} ({len(vals)})",
                edgecolor="white", linewidth=0.3)
    ax.set_xlabel(feature_labels[fi], fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.suptitle("Phân bố điểm số theo lớp — ViSP", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "hinh_4_feature_distribution_visp.png"), dpi=200)
plt.close()
print("  Saved: hinh_4_feature_distribution_visp.png")

# 4. Trọng số Logistic Regression
feat_names = ["TF-IDF", "Shingling", "Semantic", "Len Ratio"]
coefs = clf.coef_[0]
colors_coef = ["#2ecc71" if c > 0 else "#e74c3c" for c in coefs]
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(feat_names, coefs, color=colors_coef, edgecolor="white", linewidth=1.5, width=0.5)
for bar, val in zip(bars, coefs):
    y_pos = bar.get_height() + 0.02 if val > 0 else bar.get_height() - 0.02
    ax.text(bar.get_x() + bar.get_width()/2, y_pos, f"{val:.3f}",
            ha="center", va="bottom" if val > 0 else "top", fontsize=10, fontweight="bold")
ax.axhline(y=0, color="gray", linewidth=0.5)
ax.set_ylabel("Hệ số (weight)", fontsize=12)
ax.set_title("Trọng số Logistic Regression — ViSP", fontsize=13, pad=15)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "hinh_4_feature_weights_visp.png"), dpi=200)
plt.close()
print("  Saved: hinh_4_feature_weights_visp.png")

# 5. Threshold optimization
thresholds = np.arange(0.05, 0.99, 0.01)
th_f1s = []
for th in thresholds:
    y_th = (y_proba >= th).astype(int)
    th_f1s.append(precision_recall_fscore_support(y_test, y_th, average="binary")[2])
best_th = thresholds[np.argmax(th_f1s)]
best_f1_th = max(th_f1s)
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(thresholds, th_f1s, color="#9b59b6", lw=2)
ax.axvline(x=best_th, color="#e74c3c", linestyle="--", alpha=0.7,
           label=f"Threshold tối ưu = {best_th:.2f} (F1={best_f1_th*100:.2f}%)")
ax.fill_between(thresholds, th_f1s, alpha=0.1, color="#9b59b6")
ax.set_xlabel("Threshold", fontsize=12)
ax.set_ylabel("F1-Score", fontsize=12)
ax.set_title("Tối ưu Threshold — ViSP", fontsize=13, pad=15)
ax.legend(fontsize=10)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "hinh_4_threshold_optimization_visp.png"), dpi=200)
plt.close()
print(f"  Saved: hinh_4_threshold_optimization_visp.png (best_th={best_th:.2f}, F1={best_f1_th*100:.2f}%)")

# 6. Cross-validation stability
N_RUNS = 3
all_f1_cv = []
for run_seed in range(N_RUNS):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=run_seed)
    run_f1 = []
    for train_idx, test_idx in kf.split(X_train, y_train):
        X_tr, X_te = X_train[train_idx], X_train[test_idx]
        y_tr, y_te = y_train[train_idx], y_train[test_idx]
        lr = LogisticRegression(C=grid.best_params_["C"], class_weight=grid.best_params_.get("class_weight"), max_iter=2000)
        lr.fit(X_tr, y_tr)
        run_f1.append(precision_recall_fscore_support(y_te, lr.predict(X_te), average="binary")[2])
    all_f1_cv.extend(run_f1)
    print(f"  CV seed={run_seed}: F1={np.mean(run_f1)*100:.2f}% ±{np.std(run_f1)*100:.2f}%")
all_f1_cv = np.array(all_f1_cv)
fig, ax = plt.subplots(figsize=(7, 5))
bp = ax.boxplot([all_f1_cv], vert=True, patch_artist=True,
                boxprops=dict(facecolor="#3498db", alpha=0.6),
                medianprops=dict(color="red", linewidth=2),
                flierprops=dict(marker="o", markerfacecolor="#e74c3c", markersize=6))
ax.set_xticklabels(["Logistic Regression\n3 runs × 5 folds"])
ax.set_ylabel("F1-Score", fontsize=12)
ax.set_title("Cross-validation Stability — ViSP", fontsize=13, pad=15)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "hinh_4_cv_stability_visp.png"), dpi=200)
plt.close()
print(f"  Saved: hinh_4_cv_stability_visp.png (mean={np.mean(all_f1_cv)*100:.2f}%, std={np.std(all_f1_cv)*100:.2f}%)")

# === 10. SAVE MODEL ===
print("\n" + "="*60)
print("  SAVE MODEL")
print("="*60)

model_data = {
    "classifier": clf,
    "feature_names": ["tfidf", "shingling", "semantic", "len_ratio"],
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1": f1,
    "best_params": grid.best_params_,
    "cv_results": grid.cv_results_,
}
with open(os.path.join(OUTPUT_DIR, "ensemble_model_vi.pkl"), "wb") as f:
    pickle.dump(model_data, f)

shutil.make_archive("/kaggle/working/visp_trained_model", "zip", OUTPUT_DIR)
print(f"  Saved: ensemble_model_vi.pkl + tfidf_vectorizer_vi.pkl")
print(f"  Saved: phobert_finetuned/")
print(f"  Zipped: visp_trained_model.zip")
print(f"\n  {'='*50}")
print(f"  Tải các file sau từ Kaggle về:")
print(f"  - visp_trained_model.zip → giải nén → copy vào plagiarism_desktop/models/")
print(f"  - Tất cả hình trong OUTPUT_DIR → copy vào thư mục hình ảnh báo cáo")
print(f"  {'='*50}")
print(f"  Danh sách hình đã sinh:")
VISP_IMAGES = [
    "hinh_4_2_confusion_matrix_visp.png",
    "hinh_4_2_metrics_visp.png",
    "hinh_4_roc_curve_visp.png",
    "hinh_4_pr_curve_visp.png",
    "hinh_4_feature_distribution_visp.png",
    "hinh_4_feature_weights_visp.png",
    "hinh_4_threshold_optimization_visp.png",
    "hinh_4_cv_stability_visp.png",
]
for fname in VISP_IMAGES:
    fpath = os.path.join(OUTPUT_DIR, fname)
    if os.path.exists(fpath):
        print(f"  ✅ {fname}")
    else:
        print(f"  ❌ {fname}")
