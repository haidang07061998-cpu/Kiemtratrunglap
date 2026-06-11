KẾT QUẢ HUẤN LUYỆN
====================

1. PAN-PC-11 (Phát hiện đạo văn Tiếng Anh)
-------------------------------------------
Mô hình: TF-IDF + Shingling + BERT (fine-tuned) → Logistic Regression dựa trên kết hợp đa đặc trưng
Dữ liệu: PAN-PC-11 corpus (3000 cặp train: 2000 pos + 1000 neg)
Tỷ lệ split: 80-20 (fine-tune BERT trên 80% train, không data leakage)

Kết quả trên held-out val set:
  - F1:        97.37%
  - Accuracy:  96.50%
  - Precision: 98.48%
  - Recall:    96.29%

5×5 k-fold CV trên train features:
  - Logistic Regression ổn định, best params: C=0.01, class_weight=balanced

So sánh với công bố trước đây:
  - Ahuja (2019):        87.5%
  - AL-Jibory (2020):    86.7%
  - Sys-P (2017):        83.6%
  → Kết quả của chúng ta vượt trội (+9.87% so với Ahuja)

Model file: models/ensemble_model_pan.pkl


2. ViSP (Phát hiện đạo văn Tiếng Việt)
---------------------------------------
Mô hình: PhoBERT bi-encoder → cosine similarity
Dữ liệu: 308k cặp train positives + 333k cặp test positives (không overlap)

Kết quả trên held-out test set (visp_test.csv):
  - F1:        99.35%
  - Threshold: 0.63

Ghi chú: ViSP chưa có benchmark công bố trước đây để so sánh.
         Dữ liệu dễ (negative pairs ngẫu nhiên, không adversarial).

Model file: models/ensemble_model_vi.pkl


3. Phương pháp (Scoring cho ứng dụng Web)
----------------------------
Scoring: Công thức trọng số (thay vì Logistic Regression dựa trên kết hợp đa đặc trưng)
  final = Shingling×0.5 + TF-IDF×0.3 + BERT×0.2

Phân loại:
  - ≥ 70%: "Đạo văn (Copy-paste)" 🚨
  - 50-70%: "Nghi ngờ (Paraphrase)" 🔶
  - 30-50%: "Trùng chủ đề" ⚠
  - < 30%: "Không đạo văn" ✅

Cảnh báo đặc biệt: Shingling < 10% + BERT > 85% → "Trùng chủ đề (Cùng lĩnh vực, không đạo văn)"
