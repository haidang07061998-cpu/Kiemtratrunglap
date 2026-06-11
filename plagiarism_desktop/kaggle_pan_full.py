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
!pip install -q sentence-transformers scikit-learn nltk pandas numpy torch tqdm

# === 2. Imports ===
import os, pickle, re, random, shutil, warnings
import numpy as np
import pandas as pd
import nltk
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, GridSearchCV
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tqdm.notebook import tqdm

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
print("  [1/5] LOAD PAN TRAINING PAIRS")
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
print("  [2/5] TRAIN/VAL SPLIT + BERT FINE-TUNE")
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
print("  [3/5] EXTRACT FEATURES")
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
y_train = [p["label"] for p in train_pairs]
X_val = np.array([features_all(p["text_a"], p["text_b"]) for p in tqdm(val_pairs, desc="Val features")])
y_val = [p["label"] for p in val_pairs]
print(f"  Train: {X_train.shape}, Val: {X_val.shape}")

# === 7. Evaluate ===
print("\n" + "="*60)
print("  [4/5] KẾT QUẢ")
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

# === 8. Lưu ===
print("\n" + "="*60)
print("  [5/5] LƯU KẾT QUẢ")
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
print(f"  Tải file → giải nén → copy vào plagiarism_desktop/models/")
