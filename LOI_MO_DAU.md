MỞ ĐẦU

Trong bối cảnh khoa học công nghệ phát triển mạnh mẽ, trí tuệ nhân tạo đang từng bước trở thành một công cụ quan trọng trong nhiều lĩnh vực của đời sống xã hội, đặc biệt là trong giáo dục và nghiên cứu khoa học. Một trong những ứng dụng nổi bật của trí tuệ nhân tạo là xử lý ngôn ngữ tự nhiên (NLP), cho phép máy tính hiểu, phân tích và đánh giá văn bản một cách tự động, chính xác. Điều này mở ra nhiều hướng tiếp cận mới cho việc ứng dụng công nghệ vào nâng cao chất lượng quản lý học thuật và đảm bảo tính trung thực trong nghiên cứu.

Trong lĩnh vực giáo dục, đào tạo và nghiên cứu khoa học, vấn nạn đạo văn (plagiarism) đang trở thành một thách thức lớn đối với chất lượng học thuật. Đạo văn là hành vi sử dụng ý tưởng, nội dung, kết quả nghiên cứu của người khác mà không trích dẫn hoặc ghi nhận nguồn gốc một cách phù hợp. Tại Việt Nam, cùng với sự phát triển của mạng internet và kho tài liệu số, việc sao chép nội dung một cách tinh vi ngày càng gia tăng, gây khó khăn cho công tác kiểm tra, đánh giá và bảo vệ tính trung thực trong học thuật.

Các phương pháp phát hiện đạo văn truyền thống chủ yếu dựa vào so sánh chuỗi ký tự (string matching) hoặc kiểm tra thủ công của giảng viên. Những phương pháp này bộc lộ nhiều hạn chế: không phát hiện được các hình thức paraphrase tinh vi, không xử lý được khối lượng văn bản lớn, và tốn nhiều thời gian, công sức. Trong khi đó, các kỹ thuật xử lý ngôn ngữ tự nhiên hiện đại như TF-IDF, Shingling, Word Embedding và mô hình ngôn ngữ học sâu (BERT, PhoBERT) đã cho thấy tiềm năng to lớn trong việc phát hiện các dạng đạo văn phức tạp, bao gồm cả paraphrase và sao chép có chỉnh sửa.

Xuất phát từ thực tiễn đó, việc nghiên cứu và ứng dụng học máy (Machine Learning) kết hợp với kỹ thuật xử lý ngôn ngữ tự nhiên để xây dựng hệ thống phát hiện đạo văn đối với các bài báo khoa học, khóa luận và báo cáo chuyên đề mang ý nghĩa thiết thực cả về mặt khoa học lẫn ứng dụng. Đề tài không chỉ hướng đến khai thác khả năng của trí tuệ nhân tạo trong việc tự động nhận diện các dấu hiệu sao chép nội dung, mà còn góp phần nâng cao chất lượng quản lý học thuật, hỗ trợ công tác kiểm tra tính trung thực trong nghiên cứu và đào tạo.

Với tinh thần đó, nhóm em lựa chọn đề tài "Nghiên cứu, ứng dụng học máy trong phát hiện đạo văn đối với các bài báo khoa học, khóa luận và báo cáo chuyên đề dựa trên kỹ thuật xử lý ngôn ngữ tự nhiên".

* Mục tiêu và nhiệm vụ nghiên cứu

Mục tiêu tổng quát: Xây dựng hệ thống phát hiện đạo văn tự động cho các bài báo khoa học, khóa luận và báo cáo chuyên đề bằng cách kết hợp các kỹ thuật xử lý ngôn ngữ tự nhiên và học máy.

Các nhiệm vụ cụ thể:
- Nghiên cứu lý thuyết về học máy, xử lý ngôn ngữ tự nhiên và các phương pháp phát hiện đạo văn.
- Khám phá và phân tích các bộ dữ liệu: PAN-PC-11 (phát hiện đạo văn tiếng Anh) và ViSP (phát hiện đạo văn tiếng Việt).
- Xây dựng quy trình trích xuất đặc trưng văn bản bằng TF-IDF, Shingling và mô hình ngôn ngữ BERT/PhoBERT.
- Huấn luyện mô hình Logistic Regression dựa trên kỹ thuật kết hợp đa đặc trưng (Feature-level Ensemble).
- Xây dựng ứng dụng web cho phép người dùng kiểm tra mức độ trùng lặp giữa hai văn bản qua trình duyệt.
- Đánh giá hiệu quả của hệ thống trên các bộ dữ liệu chuẩn và so sánh với các nghiên cứu liên quan.

* Đối tượng và phạm vi nghiên cứu

Đối tượng nghiên cứu: Các thuật toán học máy và kỹ thuật xử lý ngôn ngữ tự nhiên trong phát hiện đạo văn, bao gồm TF-IDF, Shingling, BERT, PhoBERT và Logistic Regression.

Phạm vi nghiên cứu: Sử dụng bộ dữ liệu PAN-PC-11 (3.000 cặp văn bản tiếng Anh) để huấn luyện mô hình phát hiện đạo văn tiếng Anh và bộ dữ liệu ViSP (308.000 cặp văn bản tiếng Việt) cho phát hiện đạo văn tiếng Việt. Hệ thống được triển khai dưới dạng ứng dụng web với giao diện trực quan qua trình duyệt. Cần lưu ý rằng bộ dữ liệu ViSP có định dạng cặp câu (sentence-level paraphrase), trong khi hệ thống và pipeline xử lý có khả năng phân tách, trích xuất cấu trúc từ các văn bản dài (báo cáo, khóa luận) và áp dụng hiệu quả cho các định dạng học thuật thực tế.

* Phương pháp nghiên cứu

Phương pháp lý thuyết: Nghiên cứu tài liệu về học máy, xử lý ngôn ngữ tự nhiên, các chỉ số đánh giá (Accuracy, Precision, Recall, F1-Score). Tìm hiểu các kỹ thuật biểu diễn văn bản và mô hình ngôn ngữ học sâu.

Phương pháp thực nghiệm: Huấn luyện và đánh giá mô hình trên các bộ dữ liệu chuẩn PAN-PC-11 và ViSP. Xây dựng pipeline xử lý văn bản bao gồm tiền xử lý, trích xuất đặc trưng và phân loại. So sánh kết quả với các công bố trước đây.

* Ý nghĩa khoa học và thực tiễn

Ý nghĩa khoa học: Chuyên đề cung cấp phân tích chi tiết về việc kết hợp các kỹ thuật biểu diễn văn bản truyền thống (TF-IDF, Shingling) với mô hình ngôn ngữ học sâu (BERT, PhoBERT) trong bài toán phát hiện đạo văn. Kết quả nghiên cứu khẳng định tính hiệu quả của phương pháp kết hợp đa đặc trưng, đạt F1-Score 97,37% trên PAN-PC-11 và 99,35% trên ViSP.

Ý nghĩa thực tiễn: Hệ thống cho phép phát hiện đạo văn một cách tự động, nhanh chóng và chính xác đối với các văn bản tiếng Anh và tiếng Việt. Ứng dụng web được xây dựng giúp giảng viên, cán bộ quản lý giáo dục và nhà nghiên cứu dễ dàng kiểm tra mức độ trùng lặp của các bài báo khoa học, khóa luận và báo cáo chuyên đề qua trình duyệt, không cần cài đặt phức tạp. Về ý nghĩa xã hội, chuyên đề góp phần nâng cao chất lượng quản lý học thuật tại các trường đại học Việt Nam, hỗ trợ công tác phòng chống đạo văn và bảo vệ tính trung thực trong nghiên cứu khoa học.

* Bố cục chuyên đề

Ngoài phần mở đầu, kết luận và tài liệu tham khảo, nội dung chuyên đề gồm 4 chương:

Chương 1: Tổng quan nghiên cứu liên quan — Khảo sát tình hình nghiên cứu hiện tại về phát hiện đạo văn, bao gồm các phương pháp truyền thống, kỹ thuật học máy và học sâu, cùng các hệ thống phát hiện đạo văn phổ biến.

Chương 2: Cơ sở lý thuyết — Trình bày kiến thức nền tảng về xử lý ngôn ngữ tự nhiên, các kỹ thuật biểu diễn văn bản (TF-IDF, Shingling, Word Embedding), mô hình BERT/PhoBERT, và các chỉ số đánh giá hiệu suất.

Chương 3: Dữ liệu và thiết kế hệ thống — Phân tích hai bộ dữ liệu PAN-PC-11 và ViSP, kiến trúc hệ thống tổng thể, quy trình xử lý và xây dựng ứng dụng web.

Chương 4: Kết quả thực nghiệm — Kết quả huấn luyện mô hình, đánh giá hiệu suất, so sánh với các nghiên cứu liên quan và thảo luận về kết quả đạt được.
