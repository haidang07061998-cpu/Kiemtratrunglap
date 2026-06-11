CHƯƠNG 4: KẾT QUẢ THỰC NGHIỆM

4.1. Mô hình phát hiện đạo văn tiếng Anh (PAN-PC-11)

4.1.1. Cấu hình huấn luyện

Mô hình phát hiện đạo văn tiếng Anh được huấn luyện trên bộ dữ liệu PAN-PC-11 với cấu hình như sau:
- Bộ phân loại: Logistic Regression
- Số đặc trưng: 4 (TF-IDF similarity, Shingling similarity, Semantic similarity, Length ratio)
- Số cặp huấn luyện: 2.250 (1.500 positive + 750 negative)
- Tỷ lệ split: 80-20 (fine-tune BERT trên 80% train, không data leakage)
- Cross-validation: Stratified 5-fold
- Siêu tham số: C=0.01, class_weight="balanced", max_iter=1000
- Mô hình ngữ nghĩa: Cross-encoder hoặc SentenceTransformer (paraphrase-multilingual-mpnet-base-v2)

4.1.2. Kết quả đánh giá

*[Hình 4.1: Confusion matrix của mô hình phát hiện đạo văn tiếng Anh (PAN-PC-11)]*

Kết quả trên held-out validation set:

Chỉ số	Giá trị
F1-Score	97,37%
Accuracy	96,50%
Precision	98,48%
Recall	96,29%

Kết quả 5-fold cross-validation trên tập huấn luyện:
- Accuracy trung bình: ~95% với độ lệch chuẩn thấp, cho thấy mô hình ổn định qua các fold.
- Logistic Regression với C=0.01 và class_weight="balanced" cho kết quả tốt nhất, tránh overfitting trên dữ liệu có kích thước vừa phải.
- Các đặc trưng đóng góp tương đối đồng đều, không có hiện tượng một đặc trưng lấn át.

4.1.3. Phân tích kết quả

Precision 98,48%: Trong số các cặp bị mô hình kết luận là đạo văn, 98,48% là chính xác. Tỷ lệ false positive rất thấp, giảm thiểu các trường hợp kết luận oan sai. Recall 96,29%: Mô hình phát hiện được 96,29% tổng số cặp đạo văn thực sự. Tỷ lệ bỏ sót (false negative) chỉ 3,71%, chấp nhận được cho bài toán phát hiện đạo văn.

F1-Score 97,37%: Kết quả cân bằng giữa Precision và Recall, cho thấy mô hình hoạt động tốt trên cả hai phương diện. Đây là chỉ số tổng hợp quan trọng nhất, đặc biệt khi dữ liệu có sự mất cân bằng giữa hai lớp.

Kết quả này cho thấy sự kết hợp giữa ba phương pháp biểu diễn văn bản (TF-IDF, Shingling, BERT) là chiến lược hiệu quả. Mỗi phương pháp bổ sung cho điểm yếu của phương pháp kia: TF-IDF nắm bắt tương đồng từ vựng, Shingling phát hiện sao chép cục bộ, BERT phát hiện paraphrase ngữ nghĩa.

4.2. Mô hình phát hiện đạo văn tiếng Việt (ViSP)

4.2.1. Cấu hình huấn luyện

Mô hình phát hiện đạo văn tiếng Việt được huấn luyện trên bộ dữ liệu ViSP với cấu hình:
- Mô hình ngữ nghĩa: PhoBERT bi-encoder (vinai/phobert-base hoặc fine-tuned)
- Số cặp huấn luyện: 50.000 cặp positive + 50.000 cặp negative
- Chiến lược negative sampling: Ghép chéo ngẫu nhiên các cặp positive, shuffle seed
- Phương pháp: Bi-encoder embedding + cosine similarity
- Threshold tối ưu: 0.63

4.2.2. Kết quả đánh giá

*[Hình 4.2: Confusion matrix của mô hình phát hiện đạo văn tiếng Việt (ViSP)]*

Kết quả trên held-out test set (visp_test.csv, 333.000 cặp):

Chỉ số	Giá trị
F1-Score	99,35%
Threshold	0,63

Ghi chú: Mô hình ViSP sử dụng bi-encoder PhoBERT + cosine similarity với threshold 0.63, không qua bộ phân loại Logistic Regression. Với mỗi cặp văn bản, embedding vector được tính, cosine similarity được đo, và kết quả được so sánh với threshold để quyết định positive/negative. Từ đó có thể xây dựng confusion matrix đầy đủ (TP, TN, FP, FN) và tính các chỉ số khác. Tuy nhiên, do dữ liệu ViSP có negative pairs được sinh ngẫu nhiên (không adversarial), độ khó của bài toán thấp dẫn đến các chỉ số đều rất cao và xấp xỉ F1. Do đó, F1-Score là chỉ số đại diện đủ tin cậy. Threshold 0.63 được chọn bằng cách tối ưu F1 trên tập validation.

4.2.3. Phân tích kết quả

Với F1-Score 99,35% trên tập test 333.000 cặp, mô hình đạt hiệu suất rất cao. Threshold tối ưu 0.63 cho thấy mô hình tự tin với dự đoán của mình. Kết quả này khẳng định PhoBERT là mô hình nền tảng phù hợp cho bài toán phát hiện đạo văn tiếng Việt.

Tuy nhiên, cần lưu ý rằng negative pairs trong ViSP được sinh ngẫu nhiên từ các cặp positive khác nhau, chưa có adversarial negative samples (các cặp văn bản có chủ đề gần nhưng không phải đạo văn). Do đó, kết quả 99,35% có thể chưa phản ánh đầy đủ độ khó của bài toán trong thực tế. Trong môi trường thực tế, tồn tại nhiều trường hợp biên: hai văn bản cùng chủ đề chuyên môn, sử dụng thuật ngữ giống nhau nhưng không có hành vi sao chép. Đây là thách thức cần giải quyết trong các phiên bản sau.

4.3. So sánh với các nghiên cứu liên quan

*[Hình 4.3: Biểu đồ cột so sánh Accuracy giữa các nghiên cứu trên PAN-PC-11]*

Bảng 4.1: So sánh kết quả trên bộ dữ liệu PAN-PC-11

Nghiên cứu	Accuracy	Phương pháp
Sys-P (2017)	83,6%	n-gram + SVM
AL-Jibory & Al-Janabi (2020)	86,7%	Đặc trưng thống kê + ML
Ahuja et al. (2020)	87,5%	Hybrid TF-IDF + Shingling + SVM
Nghiên cứu này (2026)	96,50%	TF-IDF + Shingling + BERT → Logistic Regression dựa trên kết hợp đa đặc trưng

Kết quả của nhóm vượt trội so với các công bố trước đây:
- Cao hơn Ahuja (2020) 8,97 điểm phần trăm tuyệt đối.
- Cao hơn AL-Jibory (2020) 9,80 điểm phần trăm.
- Cao hơn Sys-P (2017) 12,90 điểm phần trăm.

Nguyên nhân khác biệt:
1. Kết hợp ba loại đặc trưng: Trong quá trình xây dựng, nhóm thử nghiệm từng cặp đặc trưng và nhận thấy bộ ba luôn cho kết quả tốt hơn bất kỳ tổ hợp hai chiều nào.
2. Sử dụng BERT: Mô hình ngôn ngữ học sâu cho phép phát hiện paraphrase ngữ nghĩa, điều mà TF-IDF và Shingling đơn thuần không làm được.
3. Logistic Regression kết hợp đa đặc trưng: Bộ phân loại đơn giản nhưng hiệu quả với 4 đặc trưng chất lượng cao, tránh overfitting.
4. Fine-tuning BERT trên 80% dữ liệu PAN: Giúp mô hình ngữ nghĩa thích nghi với đặc thù của bài toán phát hiện đạo văn.

Đối với tiếng Việt (ViSP), so sánh trực tiếp chưa thực hiện được do đây là bộ dữ liệu mới, chưa có benchmark công bố trước đây. Kết quả 99,35% F1 là baseline đầu tiên cho bài toán này trên thư viện ViSP.

4.4. Phân tích ảnh hưởng của từng đặc trưng (Ablation Study)

Để đánh giá đóng góp của từng đặc trưng vào hiệu suất tổng thể, chúng em tiến hành thử nghiệm ablation: huấn luyện mô hình Logistic Regression với từng đặc trưng riêng lẻ và các tổ hợp khác nhau trên bộ dữ liệu PAN-PC-11.

Bảng 4.2: Kết quả ablation study trên PAN-PC-11

Đặc trưng sử dụng	F1-Score	Accuracy
Chỉ TF-IDF	82,4%	81,1%
Chỉ Shingling	88,7%	87,3%
Chỉ Semantic (BERT)	89,1%	87,9%
TF-IDF + Shingling	91,8%	90,5%
TF-IDF + BERT	93,2%	92,0%
Shingling + BERT	94,6%	93,4%
TF-IDF + Shingling + BERT (đầy đủ)	97,37%	96,50%

Kết quả ablation study trên bảng 4.2 xác nhận trực giác ban đầu của nhóm: Shingling dù đơn giản nhưng đóng góp đáng kể hơn kỳ vọng (F1 88,7% khi đứng riêng). Điều thú vị là TF-IDF, tuy đạt thấp nhất khi đứng riêng (82,4%), lại đóng góp thêm 2,8 điểm F1 khi kết hợp với Shingling + BERT (từ 94,6% lên 97,37%). Nhóm đánh giá sự bổ sung này là minh chứng rõ ràng cho hiệu quả của feature-level ensemble: các đặc trưng yếu khi đứng riêng vẫn có thể cộng hưởng khi kết hợp đúng cách.

4.5. Phân tích chi tiết đầu ra mô hình

4.5.1. Ví dụ False Positive và False Negative

False Positive (FP): Hai văn bản cùng thuộc lĩnh vực "học máy" với các thuật ngữ giống nhau (SVM, Random Forest, cross-validation) nhưng viết về hai nghiên cứu độc lập. Shingling thấp (~5%) nhưng BERT cao (~88%), hệ thống có thể cảnh báo "Trùng chủ đề" thay vì kết luận đạo văn — cơ chế cảnh báo đặc biệt giúp giảm thiểu FP.

False Negative (FN): Hai văn bản có paraphrase mức độ cao, sử dụng hoàn toàn từ ngữ khác để diễn đạt lại cùng một ý. Shingling rất thấp (~2%), TF-IDF thấp (~30%), nhưng BERT vẫn phát hiện tương đồng (~75%). Nếu BERT dưới ngưỡng 85%, hệ thống không kích hoạt chế độ paraphrase, dẫn đến điểm tổng hợp thấp hơn thực tế. Đây là hướng cần cải thiện trong tương lai với cross-encoder mạnh hơn.

4.5.2. Thời gian inference và giới hạn xử lý

Thời gian xử lý trung bình trên ứng dụng web:
- So sánh 2 file (văn bản ~5.000 từ): 3–6 giây (bao gồm tải mô hình, trích xuất đặc trưng).
- So sánh với kho (tìm kiếm FAISS trên 100 tài liệu): 1–3 giây cho tìm kiếm + 2–5 giây cho ensemble scoring top 20.
- Xuất báo cáo PDF: 1–2 giây.

Giới hạn xử lý văn bản: Văn bản dài (khóa luận 50–100 trang, ~15.000–30.000 từ) được xử lý qua cơ chế chunking (max_words=256, overlap=50) cho tìm kiếm FAISS và phân đoạn câu cho semantic similarity. Giới hạn thực tế phụ thuộc vào bộ nhớ RAM: với 8GB RAM, hệ thống xử lý ổn định văn bản đến ~50.000 từ. Văn bản dài hơn có thể được chia nhỏ thủ công hoặc nâng cấp phần cứng.

4.6. Demo ứng dụng web

Ứng dụng web được triển khai với backend FastAPI tại http://localhost:8000. Giao diện gồm ba tab chức năng: "So sánh 2 File" cho phép kéo-thả hai văn bản và hiển thị kết quả dạng biểu đồ tròn (score gauge) cùng ba chỉ số chi tiết; "So sánh với Kho" cho phép kiểm tra một văn bản với toàn bộ kho tài liệu đã index bằng FAISS; "Quản lý Kho" cho phép tải lên, mã hóa và xóa tài liệu. Kết quả highlight được hiển thị trong modal hai cột với các đoạn trùng được tô màu (đỏ = copy, cam = paraphrase, xám = common knowledge).

*[Hình 4.4: Ảnh chụp giao diện ứng dụng web — tab "So sánh 2 File"]*
*[Hình 4.5: Ảnh chụp giao diện highlight các đoạn trùng]*

4.6.1. Quy trình phân tích 2 file

Khi người dùng chọn hai file và nhấn "BẮT ĐẦU PHÂN TÍCH", hệ thống thực hiện:
1. Đọc file A và file B bằng core/extractor.py (hỗ trợ PDF, DOCX, TXT).
2. Nhận diện ngôn ngữ tự động bằng langdetect (hoặc người dùng chọn).
3. Tính toán ba điểm số: TF-IDF similarity, Shingling similarity, Semantic similarity.
4. Kết hợp bằng công thức trọng số động.
5. Xác định các đoạn trùng cấp câu bằng cosine similarity matrix.
6. Phân loại mức độ đạo văn.
7. Hiển thị kết quả chi tiết kèm lời khuyên actionable.

Kết quả hiển thị gồm bốn mức:
- ≥ 70%: "Đạo văn (Copy-paste)" — Sao chép rõ ràng, cần xem xét.
- 50-70%: "Nghi ngờ (Paraphrase)" — Có chỉnh sửa nhưng giữ nguyên ý, cần kiểm tra.
- 30-50%: "Trùng chủ đề" — Cùng lĩnh vực, có thể không phải đạo văn.
- < 30%: "Không đạo văn" — An toàn, không đáng kể.

4.6.2. Cơ chế trọng số động

Hệ thống áp dụng hai chế độ trọng số tùy theo đặc điểm của văn bản:
- Chế độ mặc định: Shingling × 0.5 + TF-IDF × 0.3 + BERT × 0.2. Áp dụng khi không có dấu hiệu paraphrase rõ ràng, ưu tiên phát hiện sao chép nguyên văn.
- Chế độ paraphrase: BERT × 0.5 + TF-IDF × 0.3 + Shingling × 0.2. Kích hoạt khi BERT > 90% hoặc TF-IDF > 90% (dấu hiệu paraphrase có chủ đích hoặc cùng chủ đề chuyên môn), ưu tiên phát hiện tương đồng ngữ nghĩa.

4.6.3. Cảnh báo đặc biệt

Hệ thống có cơ chế cảnh báo thông minh: nếu Shingling < 10% (hai văn bản gần như không có đoạn nào trùng từ ngữ) nhưng BERT > 85% (ngữ nghĩa tương đồng cao), hệ thống đưa ra cảnh báo "Trùng chủ đề (Cùng lĩnh vực, không đạo văn)". Cơ chế này giảm thiểu false positive trong các trường hợp hai văn bản cùng thuộc một lĩnh vực chuyên môn.

4.6.4. Kết quả kiểm thử trên tập test

Hệ thống được kiểm tra trên held-out test set của bộ dữ liệu PAN-PC-11 (750 cặp, tương ứng 20% tổng số 3.000 cặp) và 333.000 cặp test của ViSP. Kết quả điển hình:
- Độ chính xác phân loại tốt: Hầu hết các cặp đạo văn bị phát hiện đúng.
- Cảnh báo phù hợp: Các cặp không đạo văn được phân loại chính xác.
- Thời gian xử lý: Khoảng 2-5 giây cho mỗi cặp văn bản (tùy thuộc vào độ dài và mô hình ngữ nghĩa).

4.7. Thảo luận kết quả và hạn chế

4.7.1. Những kết quả đạt được

Hệ thống đã đạt được những kết quả tích cực sau quá trình xây dựng và huấn luyện. Mô hình phát hiện đạo văn tiếng Anh đạt F1 97,37% trên bộ dữ liệu PAN-PC-11, vượt trội so với các công bố trước đây (Ahuja 87,5%, AL-Jibory 86,7%, Sys-P 83,6%). Mô hình phát hiện đạo văn tiếng Việt đạt F1 99,35% trên bộ dữ liệu ViSP với threshold 0,63, thiết lập baseline đầu tiên cho bài toán này.

Việc kết hợp ba phương pháp biểu diễn văn bản (TF-IDF, Shingling, BERT/PhoBERT) với cơ chế trọng số động đã chứng minh tính hiệu quả. Mỗi phương pháp bổ sung cho điểm yếu của phương pháp kia, tạo nên một hệ thống phát hiện đạo văn toàn diện.

Về mặt triển khai, ứng dụng web với backend FastAPI và frontend Bootstrap 5.3 đã được xây dựng thành công, hỗ trợ ba chế độ làm việc (so sánh 2 file, so sánh với kho tài liệu, quản lý kho). Hệ thống hỗ trợ đọc nhiều định dạng tệp (PDF, DOCX, TXT), nhận diện ngôn ngữ tự động, xuất báo cáo PDF và hiển thị highlight các đoạn trùng qua trình duyệt.

4.7.2. Hạn chế về mô hình

Mặc dù đạt kết quả cao, hệ thống vẫn tồn tại những hạn chế. Bộ dữ liệu ViSP có negative pairs được sinh ngẫu nhiên, chưa có adversarial negative samples (các cặp văn bản có chủ đề gần nhưng không đạo văn). Do đó, kết quả 99,35% có thể chưa phản ánh đầy đủ độ khó của bài toán trong thực tế.

Logistic Regression tuy đơn giản và hiệu quả nhưng chưa tận dụng được hết tiềm năng của dữ liệu. Các mô hình phức tạp hơn như XGBoost, Random Forest hoặc neural network có thể cải thiện thêm hiệu suất, đặc biệt với dữ liệu ViSP lớn.

4.7.3. Hạn chế về dữ liệu

Bộ dữ liệu PAN-PC-11 có kích thước hạn chế (3.000 cặp), có thể chưa đủ để huấn luyện mô hình deep learning phức tạp. Bộ dữ liệu ViSP tuy lớn nhưng chỉ có một nguồn dữ liệu (ViNewsQA), chưa đa dạng về phong cách viết và lĩnh vực.

4.7.4. Hướng cải tiến trong tương lai

Để khắc phục các hạn chế trên, các hướng cải tiến trong tương lai bao gồm:
- Mở rộng bộ dữ liệu với adversarial negative pairs từ các lĩnh vực chuyên môn khác nhau.
- Thử nghiệm các mô hình phức tạp hơn như XGBoost, CatBoost hoặc fine-tuning cross-encoder cho tiếng Việt.
- Tích hợp thêm đặc trưng về cấu trúc văn bản (độ dài đoạn, tỷ lệ từ loại).
- Xây dựng cơ sở dữ liệu đối chiếu lớn hơn cho tiếng Việt.
- Triển khai lên cloud (Docker + server công cộng) để nhiều người dùng có thể truy cập đồng thời thay vì chỉ chạy trên localhost.
- Tích hợp công cụ trích dẫn tự động và kiểm tra định dạng tài liệu tham khảo.
