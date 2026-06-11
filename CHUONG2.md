CHƯƠNG 2: CƠ SỞ LÝ THUYẾT

2.1. Trí tuệ nhân tạo, Học máy và Học sâu

2.1.1. Trí tuệ nhân tạo (Artificial Intelligence – AI)

Trí tuệ nhân tạo (AI) là lĩnh vực của khoa học máy tính nhằm xây dựng các hệ thống có thể thực hiện những nhiệm vụ vốn cần trí thông minh của con người như nhận thức, suy luận, học hỏi, hiểu ngôn ngữ và ra quyết định. Mục tiêu của AI không chỉ là tự động hóa mà còn hướng tới khả năng thích nghi và hỗ trợ con người trong các công việc phức tạp. Về quá trình phát triển, AI ban đầu nổi bật với các hệ chuyên gia dựa trên luật nếu - thì. Tuy nhiên, cách tiếp cận này bộc lộ hạn chế khi bài toán trở nên đa dạng và dữ liệu quá lớn. Vì vậy, AI dần chuyển sang các phương pháp dựa trên dữ liệu, trong đó học máy và học sâu giữ vai trò trung tâm.

2.1.2. Học máy (Machine Learning – ML)

Học máy (Machine Learning) là một nhánh của trí tuệ nhân tạo, trong đó máy tính học từ dữ liệu thay vì được lập trình cụ thể cho từng trường hợp. Thay vì viết quy tắc "nếu văn bản A có độ dài lớn hơn 1000 từ và có hơn 70% từ trùng khớp với văn bản B thì là đạo văn", chúng ta cho máy tính xem hàng nghìn cặp văn bản đã được gán nhãn (đạo văn hoặc không đạo văn), và máy tính tự tìm ra các quy luật để phân loại.

Các phương pháp học máy phổ biến bao gồm học có giám sát (Supervised Learning), học không giám sát (Unsupervised Learning) và học tăng cường (Reinforcement Learning). Trong bài toán phát hiện đạo văn, học có giám sát là phương pháp chủ đạo: mô hình được huấn luyện trên các cặp văn bản đã được gán nhãn "đạo văn" (positive) hoặc "không đạo văn" (negative), từ đó học cách phân loại các cặp văn bản mới.

Đầu vào của mô hình là một vector các đặc trưng số đại diện cho văn bản. Đầu ra là xác suất thuộc lớp "đạo văn" (từ 0.0 đến 1.0).

2.1.3. Học sâu (Deep Learning – DL)

Học sâu (Deep Learning) là một nhánh của học máy sử dụng mạng nơ-ron nhân tạo với nhiều lớp ẩn (deep neural networks) để học các đặc trưng phức tạp từ dữ liệu thô. Khác với học máy truyền thống yêu cầu con người thiết kế đặc trưng thủ công (feature engineering), học sâu có thể tự động học các biểu diễn đặc trưng qua nhiều tầng xử lý.

Trong bài toán phát hiện đạo văn, học sâu được ứng dụng thông qua các mô hình ngôn ngữ như BERT và PhoBERT. Các mô hình này có khả năng hiểu ngữ nghĩa sâu sắc của văn bản, cho phép phát hiện các trường hợp paraphrase tinh vi mà phương pháp truyền thống không làm được. Điểm khác biệt chính giữa học máy truyền thống và học sâu trong phát hiện đạo văn: học máy truyền thống sử dụng đặc trưng TF-IDF và Shingling do con người thiết kế, trong khi học sâu (BERT/PhoBERT) tự động học biểu diễn ngữ nghĩa từ dữ liệu văn bản thô.

2.2. Xử lý ngôn ngữ tự nhiên (Natural Language Processing – NLP)

Xử lý ngôn ngữ tự nhiên (NLP) là một nhánh của trí tuệ nhân tạo, tập trung vào việc giúp máy tính hiểu, diễn giải và sinh ra ngôn ngữ của con người. Các tác vụ NLP phổ biến bao gồm phân loại văn bản, nhận dạng thực thể, dịch máy, phân tích cảm xúc và phát hiện đạo văn.

Trong phát hiện đạo văn, NLP đóng vai trò then chốt ở nhiều khâu:
- Tiền xử lý văn bản: Loại bỏ nhiễu, chuẩn hóa, tách từ.
- Biểu diễn văn bản: Chuyển đổi văn bản thành vector số để máy tính có thể xử lý (TF-IDF, Word Embedding).
- So sánh ngữ nghĩa: Đo độ tương đồng về mặt ý nghĩa giữa hai văn bản, không chỉ dựa trên từ trùng khớp.

Các công cụ NLP thông dụng bao gồm NLTK cho tiếng Anh (tokenization, stopword removal, stemming) và UndertheSea cho tiếng Việt (word tokenization).

2.3. Kỹ thuật biểu diễn văn bản

2.3.1. TF-IDF (Term Frequency – Inverse Document Frequency)

TF-IDF là phương pháp biểu diễn văn bản phổ biến trong lĩnh vực Information Retrieval và Text Mining. TF-IDF chuyển đổi văn bản thành vector số, phản ánh tầm quan trọng của mỗi từ trong một tài liệu cụ thể so với toàn bộ tập dữ liệu.

Term Frequency (TF) đo tần suất xuất hiện của từ trong tài liệu. Một biến thể phổ biến là sublinear TF: TF(t,d) = 1 + log(f(t,d)), giúp giảm ảnh hưởng của các từ xuất hiện quá nhiều lần (áp dụng khi f(t,d) > 1). Inverse Document Frequency (IDF) đánh giá mức độ hiếm của từ trên toàn bộ corpus. Công thức sử dụng trong nghiên cứu này (tương ứng với cấu hình smooth_idf=True của thư viện scikit-learn): IDF(t) = log((1+N)/(1+DF(t))) + 1, trong đó N là tổng số tài liệu và DF(t) là số tài liệu chứa từ t. Việc cộng thêm 1 ở tử và mẫu đảm bảo không xảy ra phép chia cho 0, đồng thời mức +1 ở cuối giúp các từ xuất hiện trong mọi tài liệu vẫn có trọng số khác không.

Trọng số TF-IDF cuối cùng: TF-IDF(t,d) = TF(t,d) × IDF(t). Độ tương đồng giữa hai văn bản được tính bằng cosine similarity giữa hai vector TF-IDF: cos(θ) = (A·B) / (||A|| ||B||), trong đó A và B là hai vector TF-IDF đã được chuẩn hóa chuẩn L2.

2.3.2. Shingling

Shingling là kỹ thuật chia văn bản thành các tập hợp con (shingle) gồm k token liên tiếp. Độ tương đồng giữa hai văn bản được đo bằng chỉ số Jaccard trên hai tập shingle:

J(A, B) = |A ∩ B| / |A ∪ B|

Ví dụ, với văn bản "học máy phát hiện đạo văn", các shingle k=3 ở cấp độ từ (word-level k-gram) sẽ là: {"học máy phát", "máy phát hiện", "phát hiện đạo", "hiện đạo văn"}. Nếu văn bản khác có shingle "học máy phát" và "phát hiện đạo", độ tương đồng Jaccard sẽ là 2/(4+2-2) = 2/4 = 0.5.

Shingling có ưu điểm phát hiện chính xác các đoạn văn bản được sao chép có thay đổi nhỏ (như thêm/bớt một vài từ), vì các shingle vẫn giữ được cấu trúc cục bộ của văn bản. Một kết quả thú vị trong quá trình thử nghiệm là Shingling, dù là kỹ thuật đơn giản nhất trong ba phương pháp, lại đóng góp đáng kể vào hiệu suất tổng thể — ablation study cho thấy riêng Shingling đã đạt F1 88,7%, cao hơn TF-IDF (82,4%) và chỉ thua BERT (89,1%) không đáng kể. Điều này cho thấy đạo văn trong thực tế thường là sao chép có chỉnh sửa nhỏ chứ không phải paraphrase hoàn toàn. Phương pháp này đặc biệt hiệu quả khi kết hợp với các phương pháp biểu diễn khác, vì mỗi phương pháp bổ sung cho điểm yếu của phương pháp kia.

Khi xây dựng hệ thống, nhóm nhận thấy không có một phương pháp biểu diễn văn bản nào đủ mạnh để xử lý mọi dạng đạo văn. TF-IDF nắm bắt tốt tương đồng từ vựng nhưng bỏ qua thứ tự từ và ngữ nghĩa. Shingling phát hiện chính xác các đoạn sao chép cục bộ nhưng không hiệu quả với paraphrase dùng từ đồng nghĩa. Word Embedding và Sentence Embedding giải quyết được vấn đề ngữ nghĩa nhưng lại kém trong việc phát hiện sao chép nguyên văn. Chính vì vậy, nhóm quyết định kết hợp cả ba: TF-IDF, Shingling và BERT/PhoBERT để tận dụng điểm mạnh riêng của từng phương pháp, tạo nên một hệ thống phát hiện đạo văn toàn diện.

2.3.3. Word Embedding và Sentence Embedding

Word Embedding là kỹ thuật biểu diễn mỗi từ thành một vector số thực trong không gian nhiều chiều. Các phương pháp như Word2Vec (Mikolov et al., 2013) và FastText (Bojanowski et al., 2017) học các vector từ dựa trên ngữ cảnh xuất hiện của từ trong văn bản. Các từ có nghĩa tương đồng sẽ có vector gần nhau trong không gian embedding, cho phép phát hiện các trường hợp paraphrase sử dụng từ đồng nghĩa.

Sentence Embedding là sự phát triển tiếp theo, biểu diễn toàn bộ câu hoặc đoạn văn thành một vector duy nhất. Sentence-BERT (Reimers & Gurevych, 2019) là một trong những phương pháp phổ biến nhất, sử dụng kiến trúc siamese network để tối ưu hóa embedding sao cho các cặp văn bản tương đồng có vector gần nhau.

2.3.4. Mô hình ngôn ngữ Transformer (BERT, PhoBERT)

BERT (Bidirectional Encoder Representations from Transformers) do Devlin et al. (2019) đề xuất là một bước đột phá trong NLP. Khác với các mô hình trước đó chỉ đọc văn bản theo một chiều (từ trái sang phải hoặc từ phải sang trái), BERT sử dụng cơ chế attention hai chiều, cho phép mỗi từ "nhìn" vào tất cả các từ khác trong câu để hiểu ngữ cảnh đầy đủ.

BERT được tiền huấn luyện (pre-training) trên hai tác vụ: Masked Language Model (MLM) — dự đoán từ bị che trong câu, và Next Sentence Prediction (NSP) — dự đoán quan hệ giữa hai câu. Sau đó, mô hình có thể được tinh chỉnh (fine-tuning) cho các tác vụ cụ thể với lượng dữ liệu nhỏ hơn.

Trong phát hiện đạo văn, có hai cách sử dụng BERT:

Bi-encoder: Hai văn bản được mã hóa riêng thành hai vector embedding, sau đó tính cosine similarity giữa chúng. Cách này nhanh (có thể pre-compute embedding cho kho tài liệu), phù hợp cho tìm kiếm trên diện rộng. Tuy nhiên, do hai văn bản được mã hóa độc lập, bi-encoder không nắm bắt được tương tác chéo giữa chúng.

Cross-encoder: Hai văn bản được nối với nhau và đưa vào mô hình cùng lúc, mô hình trực tiếp tính điểm tương đồng dựa trên attention chéo giữa các token của cả hai văn bản. Cách này chính xác hơn đáng kể vì mô hình có thể so sánh trực tiếp từng cặp từ giữa hai văn bản. Tuy nhiên, cross-encoder chậm hơn (phải xử lý từng cặp riêng lẻ, không thể pre-compute) và không phù hợp cho tìm kiếm trên kho lớn. Trong hệ thống này, cross-encoder được ưu tiên cho tiếng Anh (khi có fine-tuned model), bi-encoder được dùng cho tìm kiếm FAISS và làm fallback.

PhoBERT (Nguyen & Nguyen, 2020) là phiên bản BERT dành riêng cho tiếng Việt, được tiền huấn luyện trên 20GB dữ liệu văn bản tiếng Việt (khoảng 4 tỷ token). PhoBERT sử dụng tokenization ở cấp độ âm tiết (syllable-level), phù hợp với đặc điểm của tiếng Việt là ngôn ngữ đơn lập. Embedding từ PhoBERT có thể được tính bằng mean pooling của last hidden state có trọng số theo attention mask. Do PhoBERT có giới hạn độ dài đầu vào (thường là 256 hoặc 512 token), đối với các văn bản dài như khóa luận hay báo cáo, cần áp dụng kỹ thuật phân đoạn (segmentation): văn bản được chia thành các câu hoặc đoạn nhỏ (chunk) với độ dài phù hợp, mỗi chunk được mã hóa riêng thành vector embedding, sau đó tổng hợp bằng cách lấy trung bình (mean pooling) các vector của tất cả các chunk.

2.4. Logistic Regression

Logistic Regression là thuật toán học máy có giám sát cho bài toán phân loại nhị phân. Mặc dù có tên "Regression", đây thực chất là một thuật toán phân loại. Logistic Regression ước lượng xác suất đầu vào thuộc về một lớp bằng hàm sigmoid:

P(y=1|x) = 1 / (1 + e^-(w·x + b))

Trong đó x là vector đặc trưng, w là vector trọng số và b là bias. Hàm sigmoid ánh xạ giá trị đầu ra từ (-∞, +∞) về (0, 1), cho phép giải thích kết quả dưới dạng xác suất.

Khi quyết định chọn bộ phân loại cho bài toán, nhóm đã cân nhắc giữa SVM, Random Forest và Logistic Regression. Nhóm chọn Logistic Regression vì ba lý do chính. Thứ nhất, bài toán chỉ có 4 đặc trưng đầu vào — một số lượng rất nhỏ — nên một mô hình phức tạp như Random Forest dễ bị overfitting hơn là có lợi. SVM tuy mạnh với không gian đặc trưng lớn nhưng không cho đầu ra xác suất trực tiếp, gây khó khăn khi muốn điều chỉnh ngưỡng quyết định sau này. Logistic Regression giải quyết cả hai vấn đề: nó hoạt động ổn định với số chiều thấp, và đầu ra là xác suất thực sự (0-1) thông qua hàm sigmoid, giúp nhóm dễ dàng thử nghiệm các ngưỡng khác nhau mà không cần huấn luyện lại. Thứ hai, Logistic Regression có chi phí tính toán thấp — trên máy tính cá nhân, huấn luyện chỉ mất vài giây với 2.250 cặp dữ liệu — cho phép nhóm nhanh chóng thử nghiệm các tổ hợp đặc trưng khác nhau. Thứ ba, tính diễn giải được của Logistic Regression (trọng số w cho biết mức độ ảnh hưởng của từng đặc trưng) rất hữu ích khi nhóm muốn phân tích đặc trưng nào đóng góp nhiều nhất, như ablation study ở Chương 4 sẽ cho thấy.

2.5. Học kết hợp (Ensemble Learning)

Học kết hợp (Ensemble Learning) là phương pháp kết hợp nhiều mô hình yếu (weak learners) để tạo thành một mô hình mạnh hơn (strong learner). Nguyên lý cốt lõi là tập hợp các quyết định từ nhiều mô hình khác nhau sẽ cho kết quả chính xác hơn bất kỳ mô hình đơn lẻ nào. Có hai hình thức ensemble phổ biến: feature-level ensemble (kết hợp đặc trưng từ nhiều nguồn khác nhau thành một vector đầu vào duy nhất, sau đó dùng một bộ phân loại duy nhất) và model-level ensemble (kết hợp kết quả từ nhiều mô hình độc lập bằng voting, averaging hoặc stacking). Trong bài toán phát hiện đạo văn, feature-level ensemble có thể được áp dụng để kết hợp nhiều phương pháp biểu diễn văn bản khác nhau (TF-IDF, Shingling, BERT embedding), tận dụng điểm mạnh riêng của từng phương pháp. Đây là hình thức được sử dụng trong chuyên đề này: ba đặc trưng từ ba phương pháp khác nhau được nối thành vector 4 chiều (bao gồm length ratio) làm đầu vào cho Logistic Regression, thay vì ensemble nhiều mô hình phân loại độc lập (model stacking).

2.6. Các chỉ số đánh giá mô hình

2.6.1. Ma trận nhầm lẫn (Confusion Matrix)

Ma trận nhầm lẫn là công cụ cơ bản nhất để trực quan hóa hiệu năng mô hình phân loại. Trong bài toán phân loại nhị phân phát hiện đạo văn, ma trận được biểu diễn dưới dạng bảng 2×2 với bốn giá trị:
- True Positive (TP): Cặp đạo văn được phân loại đúng là đạo văn.
- True Negative (TN): Cặp không đạo văn được phân loại đúng là không đạo văn.
- False Positive (FP): Cặp không đạo văn bị phân loại sai là đạo văn (dương tính giả).
- False Negative (FN): Cặp đạo văn bị phân loại sai là không đạo văn (âm tính giả).

2.6.2. Accuracy, Precision, Recall, F1-Score

Accuracy (Độ chính xác tổng thể): Tỷ lệ dự đoán đúng trên toàn bộ tập dữ liệu.
Accuracy = (TP + TN) / (TP + TN + FP + FN)

Precision (Độ chính xác): Trong số các cặp bị mô hình kết luận là đạo văn, bao nhiêu phần trăm là đúng.
Precision = TP / (TP + FP)

Recall (Độ bao phủ): Trong số tất cả các cặp đạo văn thực sự, bao nhiêu phần trăm được phát hiện.
Recall = TP / (TP + FN)

F1-Score: Trung bình điều hòa giữa Precision và Recall, hữu ích khi cần đánh giá cân bằng giữa hai chỉ số.
F1 = 2 × (Precision × Recall) / (Precision + Recall)

Trong phát hiện đạo văn, Precision và Recall đều quan trọng. Precision cao đảm bảo ít trường hợp bị kết luận oan sai. Recall cao đảm bảo ít trường hợp đạo văn bị bỏ sót. F1-Score là chỉ số tổng hợp được ưu tiên sử dụng, đặc biệt khi dữ liệu mất cân bằng.

2.6.3. Cross-validation

Cross-validation là kỹ thuật đánh giá mô hình bằng cách chia dữ liệu thành nhiều tập con, huấn luyện trên một phần và kiểm tra trên phần còn lại, sau đó lặp lại nhiều lần. Phương pháp này giúp đánh giá độ ổn định của mô hình và tránh overfitting. Stratified k-fold cross-validation là biến thể giữ nguyên tỷ lệ lớp trong mỗi fold, phù hợp với dữ liệu mất cân bằng. Kết quả cuối cùng là trung bình của k lần đánh giá, phản ánh độ ổn định của mô hình qua nhiều lần huấn luyện khác nhau.
