KẾT QUẢ HUẤN LUYỆN
====================

1. PAN-PC-11 (Phát hiện đạo văn Tiếng Anh)
-------------------------------------------
Mô hình: Ensemble 4 features (TF-IDF + Shingling + BERT fine-tuned + Length Ratio) → Logistic Regression
Dữ liệu: PAN-PC-11 corpus (3000 cặp train: 2000 pos + 1000 neg)
Tỷ lệ split: 80-20

Kết quả trên held-out val set:
  - F1:        97.59%
  - Accuracy:  96.83%
  - Precision: 96.97%
  - Recall:    98.21%

So sánh với công bố trước đây:
  - Ahuja (2020):        87.50%
  - AL-Jibory (2020):    86.70%
  - Sys-P (2017):        83.60%
  → Vượt Ahuja ~10% F1

Model: multilingual_bert_finetuned (fine-tuned paraphrase-multilingual-mpnet-base-v2)
Huấn luyện: Kaggle Notebook (GPU T4x2)
Model files: models/ensemble_model_pan.pkl, models/tfidf_vectorizer.pkl, models/multilingual_bert_finetuned/


2. ViSP (Phát hiện đạo văn Tiếng Việt)
---------------------------------------
Mô hình: Ensemble 4 features (TF-IDF + Shingling + PhoBERT + Length Ratio) → Logistic Regression
Dữ liệu: visp_train.csv (positives) + index-based negative sampling
Kết quả trên held-out test set:
  - F1:        99.47%
  - Accuracy:  99.46%
  - Precision: 99.04%
  - Recall:    99.90%
  - Threshold: 0.46

Ghi chú: Dữ liệu dễ (negative pairs ngẫu nhiên, không adversarial).
         Chưa có benchmark công bố trước đây để so sánh.

Huấn luyện: Kaggle Notebook (GPU T4x2)
Model files: models/ensemble_model_vi.pkl, models/tfidf_vectorizer_vi.pkl


3. Phương pháp (Scoring cho ứng dụng Web)
----------------------------
Scoring: Công thức trọng số (thay vì Logistic Regression dựa trên kết hợp đa đặc trưng)
  final = Shingling×0.5 + TF-IDF×0.3 + BERT×0.2

Phân loại:
  - ≥ 70%: "Đạo văn (Copy-paste)"
  - 50-70%: "Nghi ngờ (Paraphrase)"
  - 30-50%: "Trùng chủ đề"
  - < 30%: "Không đạo văn"

Cảnh báo đặc biệt: Shingling < 10% + BERT > 85% → "Trùng chủ đề (Cùng lĩnh vực, không đạo văn)"

Hình ảnh đánh giá: plagiarism_desktop/eval_images/ (8 biểu đồ ViSP)
