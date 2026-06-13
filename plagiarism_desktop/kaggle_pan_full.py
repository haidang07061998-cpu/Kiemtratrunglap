"""
====================================================================
  KAGGLE NOTEBOOK — PAN-PC-11 FULL PIPELINE
  TF-IDF + Shingling + BERT + Logistic Regression

  QUY TRÌNH ĐÚNG (không data leakage):
  1. Train/val split 80-20
  2. BERT fine-tune + TF-IDF fit trên 80% TRAIN
  3. Extract features riêng cho train và val
  4. Val set → F1 held-out (kết quả chính, giống lần 1 ~97%)
  5. K-fold CV trên TRAIN features → độ tin cậy của LR
====================================================================
"""

# === 1. Install ===
!pip install -q sentence-transformers scikit-learn nltk pandas numpy torch tqdm matplotlib seaborn

# === 2. Imports ===
import os, pickle, re, random, shutil, warnings
import numpy as np
import pandas as pd
import nltk
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, GridSearchCV
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tqdm.notebook import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
nltk.download("punkt", quiet=True); nltk.download("punkt_tab", quiet=True); nltk.download("stopwords", quiet=True)
STOPWORDS_EN = set(stopwords.words("english"))
STEMMER = PorterStemmer()
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# === 3. Load PAN pairs ===
print("\n" + "="*60)
print("  [1/7] LOAD PAN TRAINING PAIRS")
print("="*60)

import glob
csv_files = glob.glob("/kaggle/input/**/pan_training_pairs.csv", recursive=True)
csv_files += glob.glob("/kaggle/input/**/*.csv", recursive=True)

if csv_files:
    pan_path = csv_files[0]
    print(f"  Found: {pan_path}")
    pan_df = pd.read_csv(pan_path)
    print(f"  Loaded: {len(pan_df)} pairs")
else:
    raise FileNotFoundError("Upload pan_training_pairs.csv to Kaggle dataset!")

pairs = pan_df.to_dict("records")
pos = sum(1 for p in pairs if p["label"] == 1)
neg = sum(1 for p in pairs if p["label"] == 0)
print(f"  Pos: {pos}, Neg: {neg}")

# === 4. Preprocess ===
def preprocess_en(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = nltk.word_tokenize(text)
    tokens = [STEMMER.stem(t) for t in tokens if t not in STOPWORDS_EN and len(t) > 1]
    return " ".join(tokens)

# === 5. Train/Val split + BERT fine-tune ===
print("\n" + "="*60)
print("  [2/7] TRAIN/VAL SPLIT + BERT FINE-TUNE")
print("="*60)

train_pairs, val_pairs = train_test_split(pairs, test_size=0.2, random_state=SEED)
print(f"  Train: {len(train_pairs)}, Val: {len(val_pairs)}")

# TF-IDF vectorizer (fit trên TRAIN only)
train_texts_all = []
for p in train_pairs:
    train_texts_all.append(preprocess_en(p["text_a"]))
    train_texts_all.append(preprocess_en(p["text_b"]))
tfidf_vec = TfidfVectorizer()
tfidf_vec.fit(train_texts_all)
print(f"  TF-IDF vocab: {len(tfidf_vec.vocabulary_)}")
with open("/kaggle/working/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf_vec, f)

# BERT fine-tune trên TRAIN only
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.losses import CosineSimilarityLoss
from torch.utils.data import DataLoader

bert_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", device=DEVICE
)
train_examples = []
for p in train_pairs:
    train_examples.append(InputExample(
        texts=[p["text_a"][:2000], p["text_b"][:2000]], label=float(p["label"])
    ))
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = CosineSimilarityLoss(model=bert_model)
bert_model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=5, warmup_steps=100,
    output_path="/kaggle/working/bert_finetuned",
    show_progress_bar=True, save_best_model=True,
)
print("  BERT fine-tuned on TRAIN only!")

# === 6. Feature extraction ===
print("\n" + "="*60)
print("  [3/7] EXTRACT FEATURES")
print("="*60)

def features_all(text_a, text_b):
    pa, pb = preprocess_en(text_a), preprocess_en(text_b)
    s1 = 0.0
    if pa and pb:
        try:
            m = tfidf_vec.transform([pa, pb])
            s1 = float(cos_sim(m[0:1], m[1:2])[0][0])
        except: pass
    def sh(t, k=3):
        w = t.split()
        if len(w) < k: return {tuple(w)}
        return {tuple(w[i:i+k]) for i in range(len(w)-k+1)}
    sh1, sh2 = sh(text_a), sh(text_b)
    s2 = len(sh1 & sh2) / len(sh1 | sh2) if sh1 and sh2 else 0.0
    emb = bert_model.encode([text_a[:2000], text_b[:2000]], convert_to_numpy=True)
    s3 = float(cos_sim(emb[0:1], emb[1:2])[0][0])
    la, lb = len(text_a.split()), len(text_b.split())
    s4 = min(la, lb) / max(la, lb) if max(la, lb) > 0 else 0
    return [s1, s2, s3, s4]

X_train = np.array([features_all(p["text_a"], p["text_b"]) for p in tqdm(train_pairs, desc="Train features")])
y_train = np.array([p["label"] for p in train_pairs])
X_val = np.array([features_all(p["text_a"], p["text_b"]) for p in tqdm(val_pairs, desc="Val features")])
y_val = np.array([p["label"] for p in val_pairs])
print(f"  Train: {X_train.shape}, Val: {X_val.shape}")

# === 7. Evaluate ===
print("\n" + "="*60)
print("  [4/7] KẾT QUẢ")
print("="*60)

# GridSearch + train final LR on train set
grid = GridSearchCV(
    LogisticRegression(max_iter=2000),
    {"C": [0.01, 0.1, 1, 10], "class_weight": ["balanced", None]},
    cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
    scoring="f1", n_jobs=-1, verbose=0
)
grid.fit(X_train, y_train)
clf = grid.best_estimator_
print(f"  Best params: {grid.best_params_}")

# === KẾT QUẢ CHÍNH: Val set held-out ===
y_pred = clf.predict(X_val)
y_proba = clf.predict_proba(X_val)[:, 1]
acc = accuracy_score(y_val, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_val, y_pred, average="binary")
print(f"\n  — KẾT QUẢ HELD-OUT VAL SET (quan trọng nhất) —")
print(f"  Accuracy:  {acc*100:.2f}%")
print(f"  Precision: {prec*100:.2f}%")
print(f"  Recall:    {rec*100:.2f}%")
print(f"  F1:        {f1*100:.2f}%")

# === K-fold CV trên TRAIN features (độ tin cậy LR) ===
N_RUNS = 5
all_f1 = []
for run_seed in range(N_RUNS):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=run_seed)
    run_f1 = []
    for train_idx, test_idx in kf.split(X_train, y_train):
        X_tr, X_te = X_train[train_idx], X_train[test_idx]
        y_tr, y_te = np.array(y_train)[train_idx], np.array(y_train)[test_idx]
        lr = LogisticRegression(C=grid.best_params_["C"], class_weight=grid.best_params_["class_weight"], max_iter=2000)
        lr.fit(X_tr, y_tr)
        run_f1.append(precision_recall_fscore_support(y_te, lr.predict(X_te), average="binary")[2])
    all_f1.extend(run_f1)
    print(f"  CV seed={run_seed}: F1={np.mean(run_f1)*100:.2f}% ±{np.std(run_f1)*100:.2f}%")

print(f"\n  — K-FOLD CV TRÊN TRAIN (5 runs × 5 folds = 25) —")
print(f"  F1: {np.mean(all_f1)*100:.2f}% ±{np.std(all_f1)*100:.2f}%")

# So sánh papers
print(f"""
  {'─'*75}
  {'Metric':<20} {'BẠN (val)':<20} {'BẠN (CV)':<20} {'Ahuja':<10}
  {'─'*75}
  {'F1':<20} {f'{f1*100:.2f}%':<20} {f'{np.mean(all_f1)*100:.2f}% ±{np.std(all_f1)*100:.2f}%':<20} {'87.5%':<10}
  {'Precision':<20} {f'{prec*100:.2f}%':<20} {'—':<20} {'93.4%':<10}
  {'Recall':<20} {f'{rec*100:.2f}%':<20} {'—':<20} {'86.1%':<10}
""")

# === 8. GENERATE IMAGES FOR REPORT ===
print("\n" + "="*60)
print("  [5/7] GENERATE REPORT FIGURES")
print("="*60)

OUTPUT_DIR = "/kaggle/working/pan_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hình 4.1: Confusion Matrix
cm = confusion_matrix(y_val, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Không đạo văn", "Đạo văn"])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap="Blues", values_format="d")
ax.set_title("Hình 4.1: Confusion Matrix — Mô hình PAN-PC-11", fontsize=13, pad=15)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "hinh_4_1_confusion_matrix_pan.png"), dpi=200)
plt.close()
print("  Saved: hinh_4_1_confusion_matrix_pan.png")

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
ax.set_title("Kết quả đánh giá mô hình PAN-PC-11 (Tiếng Anh)", fontsize=13, pad=15)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "hinh_4_1_metrics_pan.png"), dpi=200)
plt.close()
print("  Saved: hinh_4_1_metrics_pan.png")

# Hình 4.3: Biểu đồ cột so sánh Accuracy với các nghiên cứu khác
fig, ax = plt.subplots(figsize=(10, 6))
studies = ["Sys-P\n(2017)", "AL-Jibory\n(2020)", "Ahuja\n(2020)", "Nghiên cứu\nnày (2026)"]
accuracies = [83.6, 86.7, 87.5, acc * 100]
colors_studies = ["#95a5a6", "#95a5a6", "#95a5a6", "#e74c3c"]
bars = ax.bar(studies, accuracies, color=colors_studies, edgecolor="white", linewidth=1.5, width=0.6)
for bar, val in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
ax.set_ylim(0, 105)
ax.set_ylabel("Accuracy (%)", fontsize=13)
ax.set_title("Hình 4.3: So sánh Accuracy trên bộ dữ liệu PAN-PC-11", fontsize=13, pad=15)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.axhline(y=acc * 100, color="#e74c3c", linestyle="--", alpha=0.3, linewidth=1)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "hinh_4_3_comparison_pan.png"), dpi=200)
plt.close()
print("  Saved: hinh_4_3_comparison_pan.png")

# === HÌNH 4.X: CÁC BIỂU ĐỒ BỔ SUNG ===

# 1. ROC Curve + AUC
fpr, tpr, _ = roc_curve(y_val, y_proba)
roc_auc = auc(fpr, tpr)
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color="#3498db", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
ax.fill_between(fpr, tpr, alpha=0.15, color="#3498db")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curve — PAN-PC-11", fontsize=13, pad=15)
ax.legend(loc="lower right", fontsize=11)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "hinh_4_roc_curve_pan.png"), dpi=200)
plt.close()
print("  Saved: hinh_4_roc_curve_pan.png")

# 2. Precision-Recall Curve
precisions, recalls, _ = precision_recall_curve(y_val, y_proba)
pr_auc = auc(recalls, precisions)
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(recalls, precisions, color="#e74c3c", lw=2, label=f"PR curve (AUC = {pr_auc:.4f})")
ax.fill_between(recalls, precisions, alpha=0.15, color="#e74c3c")
ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title("Precision-Recall Curve — PAN-PC-11", fontsize=13, pad=15)
ax.legend(loc="lower left", fontsize=11)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "hinh_4_pr_curve_pan.png"), dpi=200)
plt.close()
print("  Saved: hinh_4_pr_curve_pan.png")

# 3. Histogram điểm số theo lớp (chỉ 3 đặc trưng chính)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
feature_labels = ["TF-IDF Similarity", "Shingling Similarity", "Semantic Similarity"]
colors_hist = {0: "#e74c3c", 1: "#3498db"}
for fi in range(3):
    ax = axes[fi]
    for label in [0, 1]:
        vals = X_val[y_val == label, fi]
        ax.hist(vals, bins=30, alpha=0.6, color=colors_hist[label],
                label=f"{'Plagiarized' if label==1 else 'Non-plagiarized'} ({len(vals)})",
                edgecolor="white", linewidth=0.3)
    ax.set_xlabel(feature_labels[fi], fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.suptitle("Phân bố điểm số theo lớp — PAN-PC-11", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "hinh_4_feature_distribution_pan.png"), dpi=200)
plt.close()
print("  Saved: hinh_4_feature_distribution_pan.png")

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
ax.set_title("Trọng số Logistic Regression — PAN-PC-11", fontsize=13, pad=15)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "hinh_4_feature_weights_pan.png"), dpi=200)
plt.close()
print("  Saved: hinh_4_feature_weights_pan.png")

# 5. Threshold optimization
thresholds = np.arange(0.05, 0.99, 0.01)
th_f1s = []
for th in thresholds:
    y_th = (y_proba >= th).astype(int)
    th_f1s.append(precision_recall_fscore_support(y_val, y_th, average="binary")[2])
best_th = thresholds[np.argmax(th_f1s)]
best_f1_th = max(th_f1s)
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(thresholds, th_f1s, color="#9b59b6", lw=2)
ax.axvline(x=best_th, color="#e74c3c", linestyle="--", alpha=0.7,
           label=f"Threshold tối ưu = {best_th:.2f} (F1={best_f1_th*100:.2f}%)")
ax.fill_between(thresholds, th_f1s, alpha=0.1, color="#9b59b6")
ax.set_xlabel("Threshold", fontsize=12)
ax.set_ylabel("F1-Score", fontsize=12)
ax.set_title("Tối ưu Threshold — PAN-PC-11", fontsize=13, pad=15)
ax.legend(fontsize=10)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "hinh_4_threshold_optimization_pan.png"), dpi=200)
plt.close()
print(f"  Saved: hinh_4_threshold_optimization_pan.png (best_th={best_th:.2f}, F1={best_f1_th*100:.2f}%)")

# 6. Cross-validation stability (boxplot)
fig, ax = plt.subplots(figsize=(7, 5))
bp = ax.boxplot([all_f1], vert=True, patch_artist=True,
                boxprops=dict(facecolor="#3498db", alpha=0.6),
                medianprops=dict(color="red", linewidth=2),
                flierprops=dict(marker="o", markerfacecolor="#e74c3c", markersize=6))
ax.set_xticklabels(["Logistic Regression\n5 runs × 5 folds"])
ax.set_ylabel("F1-Score", fontsize=12)
ax.set_title("Cross-validation Stability — PAN-PC-11", fontsize=13, pad=15)
ax.set_ylim(max(0, min(all_f1) - 0.05), min(1.0, max(all_f1) + 0.05))
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "hinh_4_cv_stability_pan.png"), dpi=200)
plt.close()
print(f"  Saved: hinh_4_cv_stability_pan.png (mean={np.mean(all_f1)*100:.2f}%, std={np.std(all_f1)*100:.2f}%)")

# === 9. ABLATION STUDY (Bảng 4.2) ===
print("\n" + "="*60)
print("  [6/7] ABLATION STUDY")
print("="*60)

# Chạy ablation với 1 sub-sample để tiết kiệm thời gian
ablation_pairs, _ = train_test_split(train_pairs, test_size=0.5, random_state=SEED)
print(f"  Ablation samples: {len(ablation_pairs)}")

FEATURE_SETS = {
    "Chỉ TF-IDF":          [0],
    "Chỉ Shingling":       [1],
    "Chỉ Semantic (BERT)": [2],
    "TF-IDF + Shingling":  [0, 1],
    "TF-IDF + BERT":       [0, 2],
    "Shingling + BERT":    [1, 2],
    "TF-IDF + Shingling + BERT": [0, 1, 2],
}

ablation_X = np.array([features_all(p["text_a"], p["text_b"]) for p in tqdm(ablation_pairs, desc="Ablation features")])
ablation_y = np.array([p["label"] for p in ablation_pairs])

ablation_results = []
for name, feat_idx in FEATURE_SETS.items():
    X_sub = ablation_X[:, feat_idx]
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_f1, fold_acc = [], []
    for train_idx, test_idx in kf.split(X_sub, ablation_y):
        X_tr, X_te = X_sub[train_idx], X_sub[test_idx]
        y_tr, y_te = ablation_y[train_idx], ablation_y[test_idx]
        lr = LogisticRegression(max_iter=2000, class_weight="balanced")
        lr.fit(X_tr, y_tr)
        y_p = lr.predict(X_te)
        fold_f1.append(precision_recall_fscore_support(y_te, y_p, average="binary")[2])
        fold_acc.append(accuracy_score(y_te, y_p))
    ablation_results.append({
        "features": name,
        "f1": np.mean(fold_f1) * 100,
        "accuracy": np.mean(fold_acc) * 100,
    })
    print(f"  {name:25s} → F1: {ablation_results[-1]['f1']:.2f}%  Acc: {ablation_results[-1]['accuracy']:.2f}%")

# Vẽ biểu đồ ablation
fig, ax = plt.subplots(figsize=(10, 5))
ab_names = [r["features"] for r in ablation_results]
ab_f1 = [r["f1"] for r in ablation_results]
ab_acc = [r["accuracy"] for r in ablation_results]
x = np.arange(len(ab_names))
w = 0.35
bars1 = ax.bar(x - w/2, ab_f1, w, label="F1-Score", color="#3498db", edgecolor="white")
bars2 = ax.bar(x + w/2, ab_acc, w, label="Accuracy", color="#2ecc71", edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(ab_names, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Tỷ lệ (%)", fontsize=12)
ax.set_title("Ablation Study — Đóng góp của từng đặc trưng (PAN-PC-11)", fontsize=13, pad=15)
ax.legend(fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "hinh_4_ablation_study_pan.png"), dpi=200)
plt.close()
print("  Saved: hinh_4_ablation_study_pan.png")

# === 10. Lưu ===
print("\n" + "="*60)
print("  [7/7] LƯU KẾT QUẢ")
print("="*60)
save_dir = "/kaggle/working/trained_model"
os.makedirs(save_dir, exist_ok=True)

model_data = {
    "classifier": clf,
    "feature_names": ["tfidf", "shingling", "semantic", "len_ratio"],
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1": f1,
    "cv_f1_mean": np.mean(all_f1),
    "cv_f1_std": np.std(all_f1),
    "best_params": grid.best_params_,
    "n_runs": N_RUNS,
}
with open(os.path.join(save_dir, "ensemble_model_pan.pkl"), "wb") as f:
    pickle.dump(model_data, f)
print(f"  ensemble_model_pan.pkl saved")
shutil.copy("/kaggle/working/tfidf_vectorizer.pkl", os.path.join(save_dir, "tfidf_vectorizer.pkl"))
print(f"  tfidf_vectorizer.pkl saved")
bert_save = os.path.join(save_dir, "multilingual_bert_finetuned")
bert_model.save(bert_save)
print(f"  multilingual_bert_finetuned/ saved")
shutil.make_archive("/kaggle/working/pan_trained_model", "zip", save_dir)
print(f"  pan_trained_model.zip ready for download")
print(f"\n  {'='*50}")
print(f"  Tải các file sau từ Kaggle về:")
print(f"  - pan_trained_model.zip → giải nén → copy vào plagiarism_desktop/models/")
print(f"  - pan_figures/ (tất cả hình) → copy vào thư mục hình ảnh báo cáo")
print(f"  {'='*50}")
print(f"  Danh sách hình đã sinh trong pan_figures/:")
PAN_IMAGES = [
    "hinh_4_1_confusion_matrix_pan.png",
    "hinh_4_1_metrics_pan.png",
    "hinh_4_3_comparison_pan.png",
    "hinh_4_roc_curve_pan.png",
    "hinh_4_pr_curve_pan.png",
    "hinh_4_feature_distribution_pan.png",
    "hinh_4_feature_weights_pan.png",
    "hinh_4_threshold_optimization_pan.png",
    "hinh_4_cv_stability_pan.png",
    "hinh_4_ablation_study_pan.png",
]
for fname in PAN_IMAGES:
    if os.path.exists(os.path.join(OUTPUT_DIR, fname)):
        print(f"  ✅ {fname}")
    else:
        print(f"  ❌ {fname}")
