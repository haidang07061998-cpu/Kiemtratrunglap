CHƯƠNG 3: DỮ LIỆU VÀ THIẾT KẾ HỆ THỐNG

3.1. Bộ dữ liệu PAN-PC-11 (Phát hiện đạo văn tiếng Anh)

3.1.1. Nguồn và tổng quan

PAN-PC-11 là bộ dữ liệu chuẩn cho bài toán phát hiện đạo văn ngoại vi (external plagiarism detection), được xây dựng cho hội thảo PAN tại CLEF 2011. Bộ dữ liệu được tổ chức thành hai phần: thư mục external-detection-corpus chứa các cặp văn bản nguồn (source document) và văn bản nghi ngờ (suspicious document) cùng file chú thích XML đánh dấu chính xác vị trí và loại đạo văn.

Thư mục source-document chứa các văn bản gốc, được chia thành nhiều phần (part), mỗi văn bản có định dạng .txt. Thư mục suspicious-document chứa các văn bản nghi ngờ cũng được phân chia tương tự, kèm các file .xml chứa thông tin về các đoạn đạo văn. Mỗi file XML xác định reference đến file văn bản đáng ngờ tương ứng, danh sách các feature với type="plagiarism", bao gồm this_offset (vị trí bắt đầu trong văn bản đáng ngờ), this_length (độ dài đoạn đạo văn), source_reference (file nguồn), source_offset (vị trí trong file nguồn), source_length (độ dài trong file nguồn), và obfuscation (mức độ che giấu).

3.1.2. Thống kê và phân tích

Bộ dữ liệu PAN-PC-11 sử dụng trong dự án gồm 3.000 cặp văn bản cho huấn luyện, với tỷ lệ 2.000 positive (có đạo văn) và 1.000 negative (không đạo văn). Các cặp positive được thu thập từ file XML annotation, trong khi các cặp negative được sinh bằng cách ghép ngẫu nhiên các đoạn văn bản khác nguồn với nhau.

Thông số tổng quan:
- Tổng số cặp: 3.000 (2.000 positive + 1.000 negative)
- Định dạng: TXT cho văn bản, XML cho annotation
- Các mức obfuscation: none, random, translation, summary
- Độ dài văn bản: từ 20 ký tự đến vài nghìn ký tự
- Số lượng văn bản nguồn: đa dạng, được phân chia thành nhiều part

Quá trình trích xuất dữ liệu PAN-PC-11 được thực hiện bằng hàm collect_pan_pairs() và parse_pan_annotation() trong core/trainer.py. Hàm parse_pan_annotation() đọc file XML và trích xuất các annotation plagiarism, bao gồm offset và length trong cả văn bản đáng ngờ và văn bản nguồn. Hàm extract_pan_text_by_offset() sử dụng offset và length để cắt chính xác đoạn văn bản tương ứng, đảm bảo dữ liệu huấn luyện chính xác đến từng ký tự.

3.2. Bộ dữ liệu ViSP (Phát hiện đạo văn tiếng Việt)

3.2.1. Nguồn và tổng quan

ViSP (Vietnamese Sentence-level Paraphrase) là bộ dữ liệu phát hiện đạo văn cấp câu dành cho tiếng Việt, được xây dựng từ các nguồn báo điện tử Việt Nam. Bộ dữ liệu được lưu dưới định dạng CSV với 6 cột:

- id: Mã định danh của cặp văn bản (UUID)
- paraphrase_id: Mã định danh paraphrase (para-* cho positive, rpara-* cho negative)
- source: Nguồn dữ liệu (ví dụ: ViNewsQA)
- topic: Chủ đề (health, society, education...)
- original_text: Văn bản gốc
- paraphrase_text: Văn bản paraphrase

3.2.2. Thống kê và phân tích

Thông số tổng quan:
- Tổng số dòng train: 406.308 dòng (bao gồm cả para và rpara)
- Số cặp positive (para-*): ~308.000
- Số cặp negative (rpara-*): ~98.000 (negative ngẫu nhiên)
- Số cặp test positives: ~333.000
- Định dạng: CSV với 6 cột

Để đảm bảo không có data leakage giữa train và test, các paraphrase_id được kiểm tra chéo (overlap check). Kết quả kiểm tra cho thấy train và test hoàn toàn không có overlap, đảm bảo tính khách quan của đánh giá.

Dữ liệu bao gồm nhiều chủ đề khác nhau (health, society, education...), giúp mô hình học được đa dạng ngữ cảnh. Negative pairs được sinh bằng cách ghép ngẫu nhiên original_text của cặp này với paraphrase_text của cặp khác, tạo ra các cặp không có quan hệ ngữ nghĩa.

3.3. Kiến trúc hệ thống tổng thể

Hệ thống phát hiện đạo văn được thiết kế theo kiến trúc module hóa với ba tầng chính:

Tầng 1 — Giao diện người dùng (Frontend): Xây dựng bằng Bootstrap 5.3 kết hợp JavaScript thuần, giao tiếp với backend qua REST API, cung cấp ba chế độ làm việc:
- So sánh 2 file: So sánh trực tiếp hai văn bản để phát hiện đạo văn.
- So sánh với kho: So sánh một văn bản với toàn bộ kho tài liệu đã được index.
- Quản lý kho tài liệu: Thêm, xóa và quản lý các tài liệu trong kho.

Tầng 2 — Core logic: Xử lý trích xuất văn bản, tiền xử lý, tính độ tương đồng, phát hiện đạo văn và sinh báo cáo.

Tầng 3 — Dữ liệu: Cơ sở dữ liệu vector (FAISS) và kho tài liệu văn bản (document_store), các mô hình đã huấn luyện.

*[Hình 3.1: Sơ đồ kiến trúc hệ thống ba tầng — GUI, Core Logic, Data]*

3.4. Quy trình xử lý (Pipeline)

*[Hình 3.2: Flowchart pipeline xử lý từ đầu vào (PDF/DOCX/TXT) đến kết quả phát hiện đạo văn]*

3.4.1. Tiền xử lý văn bản (Text Preprocessing)

Tiền xử lý là bước quan trọng nhằm chuẩn hóa văn bản đầu vào trước khi trích xuất đặc trưng. Quy trình tiền xử lý được triển khai trong core/preprocessor.py với hai pipeline riêng cho tiếng Anh và tiếng Việt.

Tiền xử lý tiếng Anh: Văn bản được chuyển về chữ thường, loại bỏ ký tự đặc biệt và số, tách từ bằng NLTK word_tokenize, loại bỏ stopwords (danh sách từ dừng tiếng Anh), và thực hiện stemming bằng Porter Stemmer. Các từ có độ dài ≤ 1 ký tự cũng bị loại bỏ.

Tiền xử lý tiếng Việt: Văn bản được loại bỏ ký tự đặc biệt, chuẩn hóa khoảng trắng, và thực hiện word tokenization bằng thư viện UndertheSea (word_tokenize với format="text"). UndertheSea là thư viện xử lý ngôn ngữ tự nhiên chuyên dụng cho tiếng Việt, có khả năng nhận dạng ranh giới từ một cách chính xác.

3.4.2. Trích xuất đặc trưng (Feature Extraction)

Hệ thống trích xuất 4 đặc trưng từ mỗi cặp văn bản:

1. TF-IDF Cosine Similarity (s1): Văn bản được tiền xử lý, sau đó chuyển thành vector TF-IDF bằng TfidfVectorizer đã được huấn luyện trước. Độ tương đồng được tính bằng cosine similarity giữa hai vector. Mỗi ngôn ngữ duy trì một vectorizer riêng (tfidf_vectorizer.pkl cho tiếng Anh, tfidf_vectorizer_vi.pkl cho tiếng Việt).

2. Shingling Jaccard Similarity (s2): Văn bản được chia thành các shingle k=3 (3-gram ở cấp độ từ). Độ tương đồng Jaccard được tính bằng tỷ lệ giao trên hợp của hai tập shingle. Đặc trưng này phát hiện các đoạn sao chép nguyên văn hoặc gần nguyên văn.

3. Semantic Similarity (s3): Sử dụng mô hình ngôn ngữ học sâu để tính độ tương đồng ngữ nghĩa. Đối với tiếng Anh, ưu tiên cross-encoder (CrossEncoder fine-tuned) nếu có, fallback sang bi-encoder SentenceTransformer (paraphrase-multilingual-mpnet-base-v2). Đối với tiếng Việt, sử dụng PhoBERT bi-encoder với mean pooling của last hidden state. Quy trình fine-tune BERT/PhoBERT được thực hiện trên 80% dữ liệu huấn luyện: các cặp văn bản được tokenize và đưa vào mô hình, loss function là Binary Cross-Entropy cho cross-encoder hoặc Cosine Embedding Loss cho bi-encoder. Mô hình fine-tuned được lưu tại models/multilingual_bert_finetuned/ (tiếng Anh) và models/phobert_finetuned/ (tiếng Việt).

Trong quá trình xây dựng, nhóm gặp một số khó khăn thực tế. PhoBERT có giới hạn độ dài đầu vào 256 token, nên khi xử lý các văn bản dài (báo cáo, khóa luận), lần chạy đầu tiên bị lỗi out-of-memory (OOM). Nhóm đã khắc phục bằng cơ chế chunking: văn bản được chia thành các đoạn 256 token, overlap 50 token để đảm bảo không mất ngữ cảnh biên, mỗi chunk được mã hóa riêng, sau đó tổng hợp bằng mean pooling. Với TF-IDF, một vấn đề khác phát sinh là kích thước vectorizer rất lớn (hàng chục nghìn từ) khi huấn luyện trên toàn bộ dữ liệu PAN-PC-11, gây chậm khi tải. Nhóm đã giải quyết bằng cách giới hạn max_features=5000 và sử dụng sublinear_tf để giảm ảnh hưởng của các từ xuất hiện quá nhiều lần.

4. Length Ratio: Tỷ lệ giữa độ dài (số từ) của văn bản ngắn hơn và văn bản dài hơn, giúp phát hiện các trường hợp độ dài bất thường.

Sau khi trích xuất, bốn đặc trưng được sử dụng theo hai chế độ:
- Chế độ có mô hình: Vector [tfidf_sim, shingling_sim, semantic_sim, len_ratio] được đưa vào Logistic Regression đã huấn luyện, mô hình tự học trọng số cho từng đặc trưng qua quá trình tối ưu (weight learning, không cần đặt thủ công).
- Chế độ fallback (không có mô hình): Sử dụng công thức trọng số động thủ công: Shingling × 0.5 + TF-IDF × 0.3 + BERT × 0.2. Hai chế độ này không chạy song song; nếu có model Logistic Regression đã huấn luyện (ensemble_model_pan.pkl hoặc ensemble_model_vi.pkl), hệ thống ưu tiên dự đoán từ model. Công thức trọng số động chỉ được dùng khi chưa có model (ví dụ: lần đầu chạy ứng dụng trước khi huấn luyện).

3.4.3. Huấn luyện mô hình (Model Training)

Quy trình huấn luyện được triển khai trong core/trainer.py, chia làm hai giai đoạn:

Giai đoạn 1 — PAN-PC-11 (Tiếng Anh): Dữ liệu positive được thu thập từ file XML annotation trong PAN-PC-11 corpus (tối đa 1.500 cặp). Dữ liệu negative được sinh với tỷ lệ 0.5 so với positive (khoảng 750 cặp). Tổng số khoảng 2.250 cặp. Bốn đặc trưng được trích xuất cho mỗi cặp, tạo thành ma trận X (n_samples × 4 features). Mô hình Logistic Regression với class_weight="balanced" và max_iter=1000 được huấn luyện trên dữ liệu này.

Giai đoạn 2 — ViSP (Tiếng Việt): Dữ liệu positive (para-*) được thu thập từ visp_train.csv (tối đa 50.000 cặp). Dữ liệu negative được sinh bằng cách ghép chéo các cặp positive. Bốn đặc trưng được trích xuất tương tự, và mô hình Logistic Regression được huấn luyện.

Cross-validation: Cả hai giai đoạn sử dụng stratified 5-fold cross-validation để đánh giá. Kết quả bao gồm accuracy trung bình, F1 trung bình, độ lệch chuẩn và classification report chi tiết. Mô hình sau huấn luyện được lưu dưới dạng pickle: ensemble_model_pan.pkl và ensemble_model_vi.pkl.

3.4.4. Dự đoán và phân loại (Inference)

Quy trình dự đoán cho một cặp văn bản mới:
1. Đọc văn bản từ file (PDF, DOCX, TXT) bằng core/extractor.py.
2. Nhận diện ngôn ngữ tự động bằng langdetect (hoặc người dùng chọn thủ công).
3. Tiền xử lý văn bản theo ngôn ngữ.
4. Tính 4 đặc trưng: tfidf_sim, shingling_sim, semantic_sim, len_ratio.
5. Dự đoán xác suất đạo văn bằng Logistic Regression (nếu có model) hoặc công thức trọng số động (scoring ensemble).
6. Phân loại mức độ dựa trên ngưỡng.

3.5. Core Logic và các module

3.5.1. Module Extractor (core/extractor.py)

Module extractor chịu trách nhiệm đọc và trích xuất nội dung từ các định dạng tệp khác nhau:
- PDF: Sử dụng PyMuPDF (fitz) để đọc từng trang, hỗ trợ các tệp PDF có cấu trúc khác nhau.
- DOCX: Sử dụng python-docx để đọc nội dung từ các tệp Word.
- TXT: Đọc trực tiếp dưới dạng UTF-8.

Module còn cung cấp các tiện ích:
- detect_language(): Nhận diện ngôn ngữ của văn bản bằng langdetect (hỗ trợ tiếng Việt và tiếng Anh).
- chunk_text(): Chia văn bản dài thành các đoạn nhỏ (max_words=256, overlap=50) cho tìm kiếm FAISS.
- get_file_size_mb() và count_words(): Các hàm tiện ích thống kê.

3.5.2. Module Similarity (core/similarity.py)

Module similarity triển khai ba phương pháp tính độ tương đồng:
- compute_tfidf_similarity(): Sử dụng TfidfVectorizer từ session để chuyển đổi và tính cosine similarity.
- compute_shingling_similarity(): Tạo tập shingle k=3 và tính chỉ số Jaccard.
- compute_semantic_similarity(): Sử dụng cross-encoder (nếu có) hoặc bi-encoder SentenceTransformer/PhoBERT để tính độ tương đồng ngữ nghĩa.

Hàm compute_ensemble_score() kết hợp ba điểm số với trọng số động:
- Chế độ mặc định: Shingling × 0.5 + TF-IDF × 0.3 + BERT × 0.2
- Chế độ paraphrase: BERT × 0.5 + TF-IDF × 0.3 + Shingling × 0.2 (khi BERT > 90% hoặc TF-IDF > 90%)

Hàm classify_plagiarism() phân loại kết quả dựa trên điểm số tổng hợp:
- ≥ 95%: "Sao chép hoàn toàn (Copy-paste)"
- 50-95%: "Có chỉnh sửa (Paraphrase)"
- < 50%: "Ít trùng lặp (Không đáng kể)"

3.5.3. Module Highlighter (core/highlighter.py)

Thuật toán align câu: Module highlighter thực hiện phát hiện và đánh dấu các đoạn trùng nhau giữa hai văn bản bằng phương pháp greedy matching dựa trên ma trận tương đồng cosine:
1. Chia văn bản thành câu (dựa trên dấu câu và xuống dòng kép).
2. Lọc các câu có độ dài ≥ 8 từ (bỏ qua câu ngắn).
3. Mã hóa tất cả câu bằng mô hình ngữ nghĩa.
4. Tính ma trận tương đồng cosine giữa các câu của hai văn bản (kích thước N×M).
5. Greedy matching: duyệt từng câu bên trái, tìm câu bên phải có độ tương đồng cao nhất ≥ 0.5, gắn nhãn "copy" (≥ 0.95) hoặc "paraphrase".

Module còn triển khai bộ lọc Common Knowledge với các từ điển CNTT (cấu trúc dữ liệu và giải thuật) cho cả tiếng Việt và tiếng Anh, giúp loại trừ các trùng lặp do kiến thức phổ thông. Hàm _merge_adjacent_matches() gộp các câu trùng liên tiếp thành một đoạn, cải thiện khả năng hiển thị kết quả.

3.5.4. Module FAISS (core/faiss_manager.py)

FAISS (Facebook AI Similarity Search) được sử dụng để tìm kiếm nhanh các văn bản tương đồng trong kho tài liệu. Module FAISSManager triển khai:
- IndexFlatIP (Inner Product) với L2 normalization để tìm kiếm cosine similarity.
- Phân loại riêng cho từng ngôn ngữ (en/vi).
- Lưu index và metadata ra đĩa dưới dạng tệp .faiss và .pkl.

Khi người dùng thêm tài liệu vào kho, văn bản được chunk (256 từ, overlap 50) và mỗi chunk được mã hóa thành vector embedding, thêm vào FAISS index kèm metadata (tên file, đường dẫn). Khi tìm kiếm, hệ thống mã hóa văn bản truy vấn thành vector, tìm top_k vector gần nhất trong FAISS, trả về danh sách các tài liệu tiềm năng kèm điểm số.

3.6. Thiết kế ứng dụng Web

Ứng dụng web được xây dựng theo kiến trúc client-server: backend là API FastAPI (Python) chạy trên Uvicorn [26], frontend là ứng dụng trang đơn (SPA) xây dựng bằng Bootstrap 5.3 và JavaScript thuần.

3.6.1. Backend (FastAPI API Server)

Backend cung cấp các API endpoint cho phép frontend gọi để thực hiện các tác vụ:
- POST /api/upload: Tải lên file tạm (PDF, DOCX, TXT) để trích xuất văn bản.
- POST /api/compare-two: So sánh hai văn bản trực tiếp, trả về kết quả chi tiết.
- POST /api/compare-library: So sánh văn bản với kho tài liệu qua FAISS.
- POST /api/library/upload, /api/library/delete, /api/library/encode-all: Quản lý kho tài liệu.
- POST /api/report/export-two, /api/report/export-library: Xuất báo cáo PDF.

Các mô hình AI (TF-IDF, BERT/PhoBERT, FAISS) được tải trong luồng nền (background thread) khi ứng dụng khởi động, client kiểm tra trạng thái qua endpoint /api/health. Backend tái sử dụng toàn bộ core logic từ các module core/extractor.py, core/similarity.py, core/highlighter.py và core/faiss_manager.py.

3.6.2. Frontend (Giao diện Web)

Frontend cung cấp ba chức năng chính dưới dạng tab:
- "So sánh 2 File": Kéo-thả hoặc chọn hai file, dán văn bản trực tiếp, chọn ngôn ngữ.
- "So sánh với Kho": Chọn file và so sánh với toàn bộ kho tài liệu đã index.
- "Quản lý Kho": Xem thống kê, tải lên, mã hóa và xóa tài liệu.

Kết quả hiển thị dạng biểu đồ tròn (score gauge) và các chỉ số chi tiết. Modal highlight hiển thị hai văn bản song song với các đoạn trùng được tô màu (đỏ = copy, cam = paraphrase, xám = common knowledge).

3.6.3. So sánh với kho tài liệu

Người dùng chọn một file cần kiểm tra. Hệ thống thực hiện:
1. Nhận diện ngôn ngữ và mã hóa văn bản thành vector.
2. Tìm kiếm FAISS trong kho để lấy top 20 tài liệu tương đồng nhất.
3. Tính điểm ensemble cho từng tài liệu, sắp xếp và hiển thị top 5 nguồn trùng nhiều nhất.
4. Tính điểm trùng lặp trung bình và đưa ra đánh giá tổng thể.

Kết quả có thể được xuất ra PDF với đầy đủ thông tin và danh sách nguồn trùng.

3.7. Cơ chế cảnh báo đặc biệt và Common Knowledge Filter

Hệ thống triển khai hai cơ chế đặc biệt để tăng độ chính xác và giảm false positive:

Cảnh báo trùng chủ đề: Khi Shingling thấp (< 10%) nhưng BERT cao (> 85%), hệ thống đưa ra cảnh báo "Trùng chủ đề (Cùng lĩnh vực, không đạo văn)" thay vì kết luận đạo văn. Cơ chế này dựa trên nhận định: nếu hai văn bản không có đoạn nào trùng từ ngữ nhưng có ngữ nghĩa tương đồng cao, rất có thể chúng nói về cùng một chủ đề chuyên môn nhưng không phải đạo văn.

Common Knowledge Filter: Từ điển các thuật ngữ CNTT phổ biến (cấu trúc dữ liệu, giải thuật, độ phức tạp) được xây dựng cho cả tiếng Việt và tiếng Anh. Khi phát hiện các câu chứa common knowledge, hệ thống gắn cờ is_common_knowledge để người dùng có thể cân nhắc khi xem kết quả highlight. Điều này đặc biệt hữu ích cho khóa luận và báo cáo trong lĩnh vực CNTT.
