KẾT LUẬN

Chuyên đề đã hoàn thành việc nghiên cứu, xây dựng và đánh giá hệ thống phát hiện đạo văn đối với các bài báo khoa học, khóa luận và báo cáo chuyên đề dựa trên kỹ thuật xử lý ngôn ngữ tự nhiên và học máy, với các đóng góp chính sau:

(1) Phương pháp kết hợp đa đặc trưng: Chuyên đề đề xuất kiến trúc feature-level ensemble kết hợp ba kỹ thuật biểu diễn văn bản — TF-IDF (tương đồng từ vựng), Shingling (sao chép cục bộ) và BERT/PhoBERT (ngữ nghĩa sâu) — làm đầu vào cho bộ phân loại Logistic Regression. Kết quả ablation study cho thấy mỗi đặc trưng đóng góp riêng biệt và sự kết hợp cả ba mang lại hiệu suất vượt trội so với bất kỳ tổ hợp hai đặc trưng nào.

(2) Kết quả thực nghiệm ấn tượng: Mô hình phát hiện đạo văn tiếng Anh đạt F1 97,37% (Accuracy 96,50%, Precision 98,48%, Recall 96,29%) trên bộ dữ liệu PAN-PC-11, vượt trội so với các công bố trước đây (Ahuja 87,5%, AL-Jibory 86,7%, Sys-P 83,6%). Mô hình phát hiện đạo văn tiếng Việt đạt F1 99,35% trên bộ dữ liệu ViSP, thiết lập baseline đầu tiên cho bài toán này.

(3) Ứng dụng web thực tế: Hệ thống được triển khai dưới dạng ứng dụng web với backend FastAPI và frontend Bootstrap, cho phép so sánh trực tiếp hai văn bản hoặc so sánh với kho tài liệu đã index bằng FAISS qua trình duyệt. Hệ thống hỗ trợ đọc nhiều định dạng tệp (PDF, DOCX, TXT), nhận diện ngôn ngữ tự động, xuất báo cáo PDF và hiển thị highlight các đoạn trùng.

(4) Cơ chế xử lý thông minh: Cơ chế trọng số động và cảnh báo trùng chủ đề giúp giảm thiểu false positive trong các trường hợp cùng lĩnh vực chuyên môn. Cơ chế chunking (chia đoạn) cho phép xử lý văn bản dài đến hàng chục nghìn từ (khóa luận 50–100 trang) mà không vượt quá giới hạn độ dài đầu vào của mô hình ngôn ngữ.

Hạn chế và hướng phát triển: Bộ dữ liệu ViSP mới chỉ có negative pairs ngẫu nhiên, chưa có adversarial samples để đánh giá thực tế hơn. Văn bản dài (trên 50.000 từ) có thể gặp hạn chế về bộ nhớ khi xử lý toàn bộ. Trong tương lai, có thể mở rộng bộ dữ liệu với các cặp cùng chủ đề chuyên môn, thử nghiệm các mô hình phức tạp hơn như XGBoost hoặc cross-encoder fine-tuned cho tiếng Việt, và tối ưu hóa hiệu năng cho văn bản siêu dài.

TÀI LIỆU THAM KHẢO

Dữ liệu
[1] PAN-PC-11 Corpus, "PAN Plagiarism Corpus 2011," PAN Workshop at CLEF 2011. [Online]. Available: https://pan.webis.de/clef11/pan11-web/plagiarism-detection.html
[2] ViSP Dataset, "Vietnamese Sentence-level Paraphrase Corpus," GitHub Repository. [Online]. Available: https://github.com/TrangPhen/visp-dataset

Mô hình và thuật toán
[3] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of deep bidirectional transformers for language understanding," in Proc. NAACL-HLT, 2019, pp. 4171–4186. doi: 10.18653/v1/N19-1423
[4] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence embeddings using Siamese BERT-networks," in Proc. EMNLP-IJCNLP, 2019, pp. 3982–3992. doi: 10.18653/v1/D19-1410
[5] D. Q. Nguyen and A. T. Nguyen, "PhoBERT: Pre-trained language models for Vietnamese," in Proc. EMNLP (Findings), 2020, pp. 894–903. doi: 10.18653/v1/2020.findings-emnlp.92
[6] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011. [Online]. Available: https://scikit-learn.org/
[7] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining, 2016, pp. 785–794. doi: 10.1145/2939672.2939785
[8] A. Vaswani et al., "Attention is all you need," in Advances in Neural Information Processing Systems 30 (NeurIPS), 2017, pp. 5998–6008.
[9] T. Mikolov, I. Sutskever, K. Chen, G. S. Corrado, and J. Dean, "Distributed representations of words and phrases and their compositionality," in Advances in Neural Information Processing Systems 26 (NeurIPS), 2013, pp. 3111–3119.
[10] P. Bojanowski, E. Grave, A. Joulin, and T. Mikolov, "Enriching word vectors with subword information," Transactions of the ACL, vol. 5, pp. 135–146, 2017. doi: 10.1162/tacl_a_00051
[11] A. Z. Broder, "On the resemblance and containment of documents," in Proc. Compression and Complexity of Sequences, 1997, pp. 21–29.
[12] G. Salton and C. Buckley, "Term-weighting approaches in automatic text retrieval," Information Processing & Management, vol. 24, no. 5, pp. 513–523, 1988.

Phát hiện đạo văn
[13] R. Ahuja, S. C. S. Yadav, and S. C. Sharma, "A hybrid approach for plagiarism detection," in Proc. Int. Conf. Innovative Computing and Communications (ICICC), 2020. doi: 10.1007/978-981-15-5113-0_38
[14] F. K. AL-Jibory and A. M. Al-Janabi, "Plagiarism detection using machine learning algorithms," Journal of Engineering and Applied Sciences, vol. 15, no. 5, pp. 1272–1279, 2020.
[15] M. Potthast, B. Stein, A. Barron-Cedeno, and P. Rosso, "An evaluation framework for plagiarism detection," in Proc. 23rd Int. Conf. Computational Linguistics (COLING), 2010, pp. 997–1005.
[16] S. M. Alzahrani, N. Salim, and A. Abraham, "Understanding plagiarism linguistic patterns through natural language processing," Knowledge-Based Systems, vol. 36, pp. 121–133, 2012. doi: 10.1016/j.knosys.2012.06.009
[17] A. Barron-Cedeno and P. Rosso, "On automatic plagiarism detection based on n-gram comparison," in Proc. European Conf. Information Retrieval (ECIR), 2009, pp. 696–700.
[18] S. V. Moravvej et al., "A novel plagiarism detection approach combining BERT-based word embedding, attention-based LSTMs and an improved differential evolution algorithm," arXiv:2305.02374, 2023.
[19] C. Bouaine, F. Benabbou, and C. Zaoui, "Unlocking the potential of Transformers with mT5 and attention mechanisms in multilingual plagiarism detection," SN Computer Science, vol. 6, no. 849, 2025. doi: 10.1007/s42979-025-04379-2

Xử lý ngôn ngữ tự nhiên tiếng Việt
[20] UndertheSea Development Team, "UndertheSea: Vietnamese NLP Toolkit," GitHub Repository. [Online]. Available: https://github.com/undertheseanlp/underthesea
[21] NLTK Development Team, "Natural Language Toolkit," [Online]. Available: https://www.nltk.org/

Thư viện và công cụ
[22] J. Johnson, M. Douze, and H. Jegou, "Billion-scale similarity search with GPUs," IEEE Transactions on Big Data, vol. 7, no. 3, pp. 535–547, 2019. doi: 10.1109/TBDATA.2019.2921572
[23] PyMuPDF (fitz) Development Team, "PyMuPDF Documentation," [Online]. Available: https://pymupdf.readthedocs.io/
[24] ReportLab Development Team, "ReportLab: Open-source PDF generation," [Online]. Available: https://www.reportlab.com/
[25] FastAPI Development Team, "FastAPI: Modern, fast web framework for building APIs," [Online]. Available: https://fastapi.tiangolo.com/
[26] Uvicorn Development Team, "Uvicorn: ASGI server implementation," [Online]. Available: https://www.uvicorn.org/
[27] M. M. Danilevsky, "python-docx: Python library for creating and updating Microsoft Word files," GitHub Repository. [Online]. Available: https://github.com/python-openxml/python-docx
[28] M. Shubber, "langdetect: Language detection library ported from Google's language-detection," GitHub Repository. [Online]. Available: https://github.com/Mimino666/langdetect
