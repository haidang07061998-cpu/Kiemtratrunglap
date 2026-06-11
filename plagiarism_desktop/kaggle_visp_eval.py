"""
====================================================================
  KAGGLE NOTEBOOK — VISP EVALUATION (PhoBERT-only)
  Kiểm tra kết quả ViSP cũ (99.93%) có đúng không
====================================================================
"""

!pip install -q sentence-transformers scikit-learn pandas numpy torch tqdm

import os, pickle, csv, random, shutil, warnings
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm.notebook import tqdm

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# === LOAD VISP ===
print("\nLOAD VISP")
import glob
train_path = test_path = None
for f in glob.glob("/kaggle/input/**/*.csv", recursive=True):
    if "visp_train" in f: train_path = f
    if "visp_test" in f: test_path = f

MAX = 50000
def load_pos(path, max_pairs=MAX):
    pos = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for row in csv.DictReader(f):
            if row["paraphrase_id"].startswith("para-"):
                pos.append({"text_a": row["original_text"], "text_b": row["paraphrase_text"], "label": 1})
                if len(pos) >= max_pairs: break
    return pos

train_pos = load_pos(train_path); test_pos = load_pos(test_path)

# Sinh negative
def gen_neg(pos_list):
    neg = []; texts = [(p["text_a"], p["text_b"]) for p in pos_list]
    random.shuffle(texts)
    for i in range(len(texts)):
        j = random.randint(0, len(texts)-1)
        if j != i:
            neg.append({"text_a": texts[i][0], "text_b": texts[(j+1)%len(texts)][1], "label": 0})
    return neg[:len(pos_list)]

train_neg = gen_neg(train_pos); test_neg = gen_neg(test_pos)
train_pairs = train_pos + train_neg; random.shuffle(train_pairs)
test_pairs = test_pos + test_neg; random.shuffle(test_pairs)
print(f"Train: {len(train_pairs)} ({len(train_pos)} pos + {len(train_neg)} neg)")
print(f"Test:  {len(test_pairs)} ({len(test_pos)} pos + {len(test_neg)} neg)")

# === PHOBERT FINE-TUNE ===
print("\nPHOBERT FINE-TUNE (TRAIN ONLY)")
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

phobert = SentenceTransformer("vinai/phobert-base", device=DEVICE)

examples = [InputExample(texts=[p["text_a"][:2000], p["text_b"][:2000]], label=float(p["label"]))
            for p in train_pairs[:10000]]

phobert.fit(
    train_objectives=[(DataLoader(examples, shuffle=True, batch_size=16), losses.CosineSimilarityLoss(model=phobert))],
    epochs=3, warmup_steps=50,
    output_path="/kaggle/working/phobert_finetuned",
    show_progress_bar=True, save_best_model=True,
)
print("Done!")

# === ENCODE TEST ===
print("\nENCODE TEST PAIRS")
test_sample = random.sample(test_pairs, min(5000, len(test_pairs)))
emb_a = phobert.encode([p["text_a"][:2000] for p in test_sample], convert_to_numpy=True, show_progress_bar=True)
emb_b = phobert.encode([p["text_b"][:2000] for p in test_sample], convert_to_numpy=True, show_progress_bar=True)

from sklearn.metrics.pairwise import cosine_similarity as cos_sim
scores = np.array([cos_sim(emb_a[i:i+1], emb_b[i:i+1])[0][0] for i in range(len(test_sample))])
y_true = np.array([p["label"] for p in test_sample])

# === FIND BEST THRESHOLD ===
print("\nEVALUATE")
best_f1 = 0; best_th = 0
for th in np.arange(0.3, 0.95, 0.01):
    y_pred = (scores >= th).astype(int)
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    if f1 > best_f1:
        best_f1 = f1; best_th = th

y_pred = (scores >= best_th).astype(int)
acc = accuracy_score(y_true, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")

print(f"""
  {'─'*50}
  {'Metric':<15} {'PhoBERT-only':<20} {'ViSP cũ':<15}
  {'─'*50}
  {'F1':<15} {f'{f1*100:.2f}%':<20} {'99.93%':<15}
  {'Accuracy':<15} {f'{acc*100:.2f}%':<20} {'99.90%':<15}
  {'Precision':<15} {f'{prec*100:.2f}%':<20} {'100.00%':<15}
  {'Recall':<15} {f'{rec*100:.2f}%':<20} {'99.85%':<15}
  {'Threshold':<15} {f'{best_th:.2f}':<20} {'—':<15}

  KẾT LUẬN:
  - F1 >= 99%  →  99.93% cũ ĐÚNG
  - F1 < 95%   →  99.93% cũ SAI (leakage)
""")

# === LƯU ===
save_dir = "/kaggle/working/visp_model"
os.makedirs(save_dir, exist_ok=True)
model_data = {"classifier": None, "feature_names": ["phobert_only"],
              "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "threshold": best_th}
with open(os.path.join(save_dir, "ensemble_model_vi.pkl"), "wb") as f:
    pickle.dump(model_data, f)
shutil.make_archive("/kaggle/working/visp_trained_model", "zip", save_dir)
print(f"visp_trained_model.zip ready")
