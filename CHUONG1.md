CHƯƠNG 1: TỔNG QUAN NGHIÊN CỨU LIÊN QUAN

1.1 Giới thiệu

Chương này khảo sát tình hình nghiên cứu hiện tại về phát hiện đạo văn (plagiarism detection), bao gồm các phương pháp truyền thống, kỹ thuật học máy và học sâu, các hệ thống thương mại và mã nguồn mở, cùng các công trình học thuật sử dụng các bộ dữ liệu chuẩn.

1.2 Các phương pháp phát hiện đạo văn truyền thống

1.2.1 Phương pháp so sánh chuỗi (String Matching)

Phương pháp phát hiện đạo văn sớm nhất dựa trên việc so sánh trực tiếp chuỗi ký tự giữa hai văn bản. Thuật toán so khớp chuỗi con dài nhất (Longest Common Subsequence - LCS) và khoảng cách Levenshtein (Levenshtein distance) được sử dụng để đo độ tương đồng giữa các văn bản. Các hệ thống như MOSS (Measure of Software Similarity) do Aiken (1994) phát triển tại Đại học Stanford sử dụng kỹ thuật winnowing để phát hiện đạo văn trong mã nguồn phần mềm.

Phương pháp này có ưu điểm là đơn giản, dễ triển khai và có độ chính xác cao với các trường hợp sao chép nguyên văn (copy-paste). Tuy nhiên, nhược điểm lớn nhất là không phát hiện được các hình thức đạo văn có chỉnh sửa như paraphrase, thay đổi từ đồng nghĩa hoặc thay đổi cấu trúc câu. Đây là lý do các phương pháp dựa trên biểu diễn ngữ nghĩa ra đời để khắc phục hạn chế này.

1.2.2 Phương pháp dựa trên Shingling và n-gram

Shingling là kỹ thuật chia văn bản thành các tập hợp con (shingle) gồm k token liên tiếp, sau đó so sánh mức độ giao nhau giữa hai tập hợp. Broder (1997) giới thiệu kỹ thuật shingling kết hợp với MinHash để ước lượng nhanh độ tương đồng Jaccard giữa các văn bản lớn. Phương pháp này đặc biệt hiệu quả trong phát hiện sao chép một phần văn bản, vì các shingle có thể phát hiện các đoạn trùng nhau ngay cả khi chúng nằm ở vị trí khác nhau trong văn bản.

Công thức độ tương đồng Jaccard trên các tập shingle được định nghĩa như sau:

J(A, B) = |A ∩ B| / |A ∪ B|

Shingling đặc biệt hiệu quả trong phát hiện các trường hợp sao chép có chỉnh sửa nhỏ, như thay đổi một số từ nhưng giữ nguyên cấu trúc câu.

1.2.3 Phương pháp dựa trên TF-IDF

TF-IDF (Term Frequency - Inverse Document Frequency) là kỹ thuật biểu diễn văn bản dưới dạng vector trọng số, trong đó mỗi từ được gán trọng số tỷ lệ thuận với tần suất xuất hiện trong văn bản và tỷ lệ nghịch với tần suất xuất hiện trong toàn bộ tập văn bản. Salton & Buckley (1988) đề xuất mô hình không gian vector (Vector Space Model) kết hợp TF-IDF, cho phép tính độ tương đồng cosine giữa các văn bản.

Phương pháp TF-IDF có khả năng phát hiện các đoạn văn bản chia sẻ nhiều từ khóa quan trọng, ngay cả khi thứ tự từ bị thay đổi. Đây là bước tiến đáng kể so với so sánh chuỗi đơn thuần. TF-IDF thường được sử dụng làm một đặc trưng thành phần trong các hệ thống phát hiện đạo văn, giúp nắm bắt sự tương đồng về mặt từ vựng giữa hai văn bản. Tuy nhiên, TF-IDF có nhược điểm là không giữ được thứ tự của từ trong câu (mô hình túi từ - Bag of Words) và hoàn toàn bỏ qua ngữ nghĩa của từ, dẫn đến hạn chế trong việc phát hiện các trường hợp paraphrase sử dụng từ đồng nghĩa hoặc cấu trúc câu khác. Đây là động lực để các phương pháp Word Embedding và mô hình ngôn ngữ học sâu ra đời, được trình bày ở mục 1.3.2 và 1.3.3.

1.3 Các phương pháp phát hiện đạo văn dựa trên học máy và học sâu

1.3.1 Học máy truyền thống

Các thuật toán học máy như Logistic Regression, Support Vector Machine (SVM), Random Forest, XGBoost [7] và Gradient Boosting đã được ứng dụng rộng rãi trong bài toán phát hiện đạo văn. Các phương pháp này thường kết hợp nhiều đặc trưng khác nhau từ văn bản, bao gồm độ tương đồng TF-IDF, độ tương đồng shingling, độ dài văn bản, tỷ lệ từ trùng, và các đặc trưng thống kê khác.

Ahuja et al. (2020) đề xuất phương pháp phát hiện đạo văn hybrid kết hợp nhiều kỹ thuật biểu diễn văn bản, đạt 87,5% accuracy trên tập dữ liệu PAN-PC-11. Công trình này sử dụng các đặc trưng từ TF-IDF và n-gram kết hợp với bộ phân loại SVM, là một trong những nghiên cứu có độ chính xác cao nhất tại thời điểm công bố.

AL-Jibory & Al-Janabi (2020) sử dụng kỹ thuật kết hợp giữa các đặc trưng thống kê và học máy truyền thống, đạt 86,7% trên PAN-PC-11. Sys-P (2017) đạt 83,6% với phương pháp dựa trên n-gram và SVM. Các kết quả này cho thấy tiềm năng của học máy trong phát hiện đạo văn, đồng thời đặt ra thách thức về việc cải thiện độ chính xác để đáp ứng yêu cầu thực tế.
1.3.2 Word Embedding và Sentence Embedding

Word embedding như Word2Vec (Mikolov et al., 2013), GloVe (Pennington et al., 2014) và FastText (Bojanowski et al., 2017) cho phép biểu diễn từ dưới dạng vector số thực, nắm bắt quan hệ ngữ nghĩa giữa các từ. Các từ có nghĩa tương đồng sẽ có vector gần nhau trong không gian embedding. Kỹ thuật này giúp phát hiện các trường hợp paraphrase sử dụng từ đồng nghĩa, một dạng đạo văn khó phát hiện bằng phương pháp truyền thống.

Sentence embedding là sự phát triển tiếp theo, biểu diễn toàn bộ câu hoặc đoạn văn thành một vector duy nhất. Các mô hình như Sentence-BERT (Reimers & Gurevych, 2019) tối ưu hóa quá trình này, cho phép tính độ tương đồng ngữ nghĩa giữa các câu một cách hiệu quả. Phương pháp này đặc biệt hữu ích trong phát hiện đạo văn dạng paraphrase, nơi nội dung được diễn đạt lại hoàn toàn bằng ngôn ngữ khác nhưng giữ nguyên ý nghĩa.

1.3.3 Mô hình ngôn ngữ Transformer (BERT, PhoBERT)

BERT (Bidirectional Encoder Representations from Transformers) do Devlin et al. (2019) đề xuất đã tạo nên bước đột phá trong xử lý ngôn ngữ tự nhiên. Với kiến trúc Transformer hai chiều, BERT có khả năng hiểu ngữ cảnh của từ dựa trên cả thông tin bên trái và bên phải. Mô hình này được tiền huấn luyện (pre-training) trên dữ liệu văn bản khổng lồ và có thể được tinh chỉnh (fine-tuning) cho các tác vụ cụ thể, trong đó có phát hiện đạo văn.

Cross-Encoder là một biến thể của BERT, trong đó hai văn bản được đưa vào mô hình đồng thời để tính điểm tương đồng trực tiếp. Phương pháp này cho độ chính xác cao hơn bi-encoder nhưng chi phí tính toán lớn hơn.

PhoBERT (Nguyen & Nguyen, 2020) là mô hình ngôn ngữ tiền huấn luyện dành riêng cho tiếng Việt, dựa trên kiến trúc BERT. PhoBERT đạt kết quả state-of-the-art trên nhiều tác vụ NLP tiếng Việt, bao gồm phân loại văn bản, nhận dạng thực thể và trích xuất quan hệ.

Các nghiên cứu gần đây (2023–2025) tiếp tục mở rộng hướng tiếp cận này. Moravvej et al. (2023) đề xuất mô hình BPD-IDE kết hợp BERT embedding với LSTM có cơ chế attention và thuật toán Differential Evolution cải tiến, đạt kết quả vượt trội trên các bộ dữ liệu MSRP, SNLI và SemEval2014 [18]. Bouaine et al. (2025) sử dụng mT5 kết hợp multi-head attention cho phát hiện đạo văn đa ngôn ngữ, đạt PlagDet 98,73% trên cặp Anh-Đức [19]. Các hướng nghiên cứu này cho thấy xu hướng kết hợp mô hình ngôn ngữ lớn với cơ chế attention và feature engineering để nâng cao độ chính xác.

1.4 Các hệ thống phát hiện đạo văn phổ biến

1.4.1 Hệ thống thương mại

Turnitin (https://www.turnitin.com/) là hệ thống phát hiện đạo văn thương mại phổ biến nhất hiện nay, được sử dụng bởi hơn 15.000 tổ chức giáo dục trên toàn thế giới. Turnitin so sánh văn bản với cơ sở dữ liệu khổng lồ gồm hơn 90 tỷ trang web, 70 triệu bài báo và hơn 380 triệu bài luận của sinh viên. Hệ thống này đạt độ chính xác cao với các trường hợp sao chép nguyên văn nhưng còn hạn chế trong việc phát hiện paraphrase tinh vi.

Grammarly (https://www.grammarly.com/) cung cấp tính năng phát hiện đạo văn tích hợp trong trình kiểm tra ngữ pháp, với cơ sở dữ liệu gồm 16 tỷ trang web. Dịch vụ này phù hợp với kiểm tra nhanh nhưng không chuyên sâu như Turnitin.

PlagScan (nay là Ouriginal) và Unicheck là các hệ thống thương mại khác, sử dụng kết hợp so sánh văn bản, cơ sở dữ liệu đối chiếu và các kỹ thuật phát hiện tinh vi. Tuy nhiên, các hệ thống này chủ yếu tập trung vào tiếng Anh và một số ngôn ngữ châu Âu, hỗ trợ tiếng Việt còn hạn chế.

1.4.2 Hệ thống mã nguồn mở

MOSS (Measure Of Software Similarity) là công cụ mã nguồn mở của Đại học Stanford, chuyên phát hiện đạo văn trong mã nguồn phần mềm. MOSS sử dụng kỹ thuật winnowing với các hash để so sánh hiệu quả các tệp mã nguồn lớn. Công cụ này được sử dụng rộng rãi trong các khóa học lập trình tại nhiều trường đại học.

JPlag (Prechelt et al., 2002) là công cụ phát hiện đạo văn trong mã nguồn và văn bản tự nhiên. JPlag sử dụng kỹ thuật so sánh cấu trúc và greedy string tiling để phát hiện sao chép. Công cụ này hỗ trợ nhiều ngôn ngữ lập trình và có giao diện web.

PLAG (System for Plagiarism Detection) được giới thiệu tại PAN competition, sử dụng các kỹ thuật xử lý ngôn ngữ tự nhiên và học máy để phát hiện đạo văn ngoại vi (external plagiarism) và đạo văn nội vi (intrinsic plagiarism). Các hệ thống tham gia PAN competition thường đạt độ chính xác từ 70-90% trên bộ dữ liệu PAN-PC-11.

1.5 Các nghiên cứu sử dụng bộ dữ liệu PAN-PC-11

PAN-PC-11 là bộ dữ liệu chuẩn cho bài toán phát hiện đạo văn, được xây dựng cho hội thảo PAN (Uncovering Plagiarism, Authorship, and Social Software Misuse). Bộ dữ liệu gồm các cặp văn bản tiếng Anh được gán nhãn "đạo văn" hoặc "không đạo văn", với các mức độ che giấu khác nhau từ sao chép nguyên văn đến paraphrase nâng cao.

Bộ dữ liệu PAN-PC-11 gồm 3.000 cặp văn bản cho huấn luyện (2.000 positive + 1.000 negative) và một tập kiểm tra riêng. Dữ liệu bao gồm cả trường hợp nguồn và văn bản đáng ngờ nằm cùng chủ đề nhưng không có đạo văn, tạo thách thức cho các hệ thống phát hiện.

Các nghiên cứu tiêu biểu trên PAN-PC-11:
- Ahuja et al. (2020): 87,5% — Phương pháp hybrid TF-IDF + Shingling + SVM
- AL-Jibory & Al-Janabi (2020): 86,7% — Đặc trưng thống kê + học máy
- Sys-P (2017): 83,6% — n-gram + SVM
- Nghiên cứu này: 97,37% F1 (96,50% Accuracy) — TF-IDF + Shingling + BERT → Logistic Regression dựa trên kết hợp đa đặc trưng

Kết quả của chúng em vượt trội so với các công bố trước đây (+9,87% so với Ahuja), nhờ vào việc kết hợp ba loại đặc trưng bổ sung cho nhau: TF-IDF nắm bắt tương đồng từ vựng, Shingling phát hiện sao chép cục bộ và Semantic similarity (BERT) phát hiện paraphrase ngữ nghĩa.

Bộ dữ liệu ViSP (Vietnamese Sentence-level Paraphrase) là bộ dữ liệu phát hiện đạo văn cấp câu cho tiếng Việt. ViSP gồm 308.000 cặp train positives và 333.000 cặp test positives, được xây dựng bằng cách crawl dữ liệu từ các báo tiếng Việt và tạo paraphrase tự động. Các cặp train và test được đảm bảo không overlap (kiểm tra bằng paraphrase_id) để tránh data leakage. Do là bộ dữ liệu mới, ViSP chưa có benchmark công bố trước đây để so sánh. Dự án của chúng em đạt F1 99,35% với threshold 0.63 trên tập test.

1.6 So sánh với các phương pháp trên

Bảng 1.1 dưới đây tổng hợp các công trình nghiên cứu liên quan đến phát hiện đạo văn:

Bảng 1.1: Tổng hợp các nghiên cứu liên quan

Công trình	Phương pháp	Bộ dữ liệu	Accuracy	F1-Score
Ahuja et al. (2020)	TF-IDF + Shingling + SVM	PAN-PC-11	87,5%	—
AL-Jibory & Al-Janabi (2020)	Đặc trưng thống kê + ML	PAN-PC-11	86,7%	—
Sys-P (2017)	n-gram + SVM	PAN-PC-11	83,6%	—
Nghiên cứu này (2026)	TF-IDF + Shingling + BERT → Logistic Regression dựa trên kết hợp đa đặc trưng	PAN-PC-11 + ViSP	96,50% (PAN)	97,37% (PAN), 99,35% (VI)
Turnitin	Đối chiếu CSDL + String matching	CSDL nội bộ	—	—
MOSS (Aiken, 1994)	Winnowing	Mã nguồn	—	—

Các nghiên cứu liên quan chủ yếu tập trung vào phát hiện đạo văn tiếng Anh trên bộ dữ liệu PAN-PC-11. Kết quả của nghiên cứu này vượt trội so với các công bố trước đây (cao hơn Ahuja 9 điểm phần trăm tuyệt đối), nhờ vào việc kết hợp ba loại đặc trưng bổ sung cho nhau: TF-IDF nắm bắt tương đồng từ vựng ở cấp độ toàn văn bản, Shingling phát hiện sao chép cục bộ từ 3 từ trở lên, và Semantic similarity (BERT) phát hiện paraphrase ngữ nghĩa mà hai phương pháp kia không làm được. Sự bổ sung này là yếu tố then chốt giúp mô hình đạt hiệu suất cao hơn các phương pháp chỉ dùng một hoặc hai đặc trưng.
