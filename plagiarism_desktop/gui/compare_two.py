import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from core.extractor import extract_text, detect_language, get_file_size_mb, count_words
from core.similarity import compute_ensemble_score, classify_plagiarism
from core.highlighter import find_matching_positions, determine_match_type
from core.report_generator import generate_pairwise_report
from gui.highlight_viewer import HighlightViewerWindow

REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")


def _score_label(v):
    """Trả về nhãn mức độ cho 1 giá trị 0-1 (chuẩn hóa)"""
    if v > 0.80:
        return "Cực kỳ cao"
    elif v > 0.50:
        return "Cao"
    elif v > 0.20:
        return "Trung bình"
    else:
        return "Thấp"


class CompareTwoWindow:
    def __init__(self, parent, session):
        self.session = session
        self.window = tk.Toplevel(parent)
        self.window.title("SO SÁNH 2 FILE")
        self.window.geometry("650x620")
        self.window.minsize(600, 500)

        self.file_a_path = None
        self.file_b_path = None
        self.result_data = None

        self._build_ui()

    def _build_ui(self):
        main = ttk.Frame(self.window)
        main.pack(fill=tk.BOTH, expand=True)

        # File A
        f1 = ttk.Frame(main)
        f1.pack(fill=tk.X, pady=2, padx=10)
        ttk.Label(f1, text="File A:").pack(side=tk.LEFT)
        self.entry_a = ttk.Entry(f1)
        self.entry_a.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(f1, text="CHỌN FILE", command=self._choose_file_a).pack(side=tk.RIGHT)

        # File B
        f2 = ttk.Frame(main)
        f2.pack(fill=tk.X, pady=2, padx=10)
        ttk.Label(f2, text="File B:").pack(side=tk.LEFT)
        self.entry_b = ttk.Entry(f2)
        self.entry_b.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(f2, text="CHỌN FILE", command=self._choose_file_b).pack(side=tk.RIGHT)

        # Language
        lf = ttk.LabelFrame(main, text="NGÔN NGỮ", padding=8)
        lf.pack(fill=tk.X, padx=10, pady=5)
        self.lang_var = tk.StringVar(self.window, value="auto")
        ttk.Radiobutton(lf, text="Tự động", variable=self.lang_var, value="auto").pack(side=tk.LEFT, padx=6)
        ttk.Radiobutton(lf, text="Tiếng Anh", variable=self.lang_var, value="en").pack(side=tk.LEFT, padx=6)
        ttk.Radiobutton(lf, text="Tiếng Việt", variable=self.lang_var, value="vi").pack(side=tk.LEFT, padx=6)

        # Analyze button
        self.btn_analyze = ttk.Button(main, text="BẮT ĐẦU PHÂN TÍCH", command=self._start_analysis)
        self.btn_analyze.pack(pady=8)

        # Progress
        self.progress = ttk.Progressbar(main, mode="determinate")
        self.progress.pack(fill=tk.X, padx=10, pady=2)
        self.status_label = ttk.Label(main, text="")
        self.status_label.pack()

        # Result
        rf = ttk.LabelFrame(main, text="KẾT QUẢ PHÂN TÍCH ĐẠO VĂN", padding=8)
        rf.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.result_text = tk.Text(rf, state=tk.DISABLED, wrap=tk.WORD,
                                   font=("Consolas", 10))
        self.result_text.pack(fill=tk.BOTH, expand=True)

        # Detail toggle — bấm để xem TF-IDF / Shingling / BERT
        self.detail_btn = ttk.Button(
            rf, text="[ Xem chi tiết thuật toán (TF-IDF, Shingling, BERT) ]",
            command=self._toggle_detail
        )
        self.detail_btn.pack(pady=(2, 0))

        self.detail_frame = ttk.Frame(rf)
        self.detail_visible = False

        self.detail_text = tk.Text(self.detail_frame, height=5, state=tk.DISABLED,
                                   font=("Consolas", 9))
        self.detail_text.pack(fill=tk.BOTH, expand=True)

        # Bottom buttons
        bf = ttk.Frame(main)
        bf.pack(pady=5)
        ttk.Button(bf, text="📄 XUẤT PDF", command=self._export_pdf).pack(side=tk.LEFT, padx=5)
        ttk.Button(bf, text="🔍 XEM HIGHLIGHT", command=self._view_highlight).pack(side=tk.LEFT, padx=5)

    def _choose_file_a(self):
        path = filedialog.askopenfilename(
            filetypes=[("Documents", "*.pdf;*.docx;*.txt")]
        )
        if path:
            self.file_a_path = path
            self.entry_a.delete(0, tk.END)
            self.entry_a.insert(0, os.path.basename(path))

    def _choose_file_b(self):
        path = filedialog.askopenfilename(
            filetypes=[("Documents", "*.pdf;*.docx;*.txt")]
        )
        if path:
            self.file_b_path = path
            self.entry_b.delete(0, tk.END)
            self.entry_b.insert(0, os.path.basename(path))

    def _start_analysis(self):
        if not self.file_a_path or not self.file_b_path:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn cả 2 file!")
            return

        self.btn_analyze.config(state=tk.DISABLED)
        self.progress["value"] = 0
        self.status_label.config(text="Đang phân tích...")

        thread = threading.Thread(target=self._run_analysis, daemon=True)
        thread.start()

    def _run_analysis(self):
        try:
            self._update_status(10, "Đang đọc File A...")
            text_a = extract_text(self.file_a_path)

            self._update_status(25, "Đang đọc File B...")
            text_b = extract_text(self.file_b_path)

            lang = self.lang_var.get()
            if lang == "auto":
                lang_a = detect_language(text_a)
                lang_b = detect_language(text_b)
                lang = lang_a if lang_a == lang_b else "en"

            self._update_status(40, "Đang tính điểm tương đồng...")
            final_score, s1, s2, s3 = compute_ensemble_score(
                text_a, text_b, lang, self.session
            )

            self._update_status(65, "Đang tìm đoạn trùng...")
            matches = find_matching_positions(
                text_a, text_b, lang, self.session
            )

            self._update_status(85, "Đang tổng hợp kết quả...")
            level = classify_plagiarism(final_score, s1, s2, s3)

            # Xác định chế độ trọng số đã dùng
            if s3 > 0.9 or s1 > 0.9:
                weight_mode = "Chế độ Paraphrase: Ngữ nghĩa×0.5 + Từ khóa×0.3 + Nguyên văn×0.2"
            else:
                weight_mode = "Chế độ mặc định: Nguyên văn×0.5 + Từ khóa×0.3 + Ngữ nghĩa×0.2"

            self.result_data = {
                "final_score": final_score,
                "score_1": s1,
                "score_2": s2,
                "score_3": s3,
                "level": level,
                "matches": matches,
                "file_a": os.path.basename(self.file_a_path),
                "file_b": os.path.basename(self.file_b_path),
                "file_a_text": text_a,
                "file_b_text": text_b,
                "language": lang,
                "weight_mode": weight_mode,
            }

            self.window.after(0, lambda: self._display_result(final_score, s1, s2, s3, level, matches, weight_mode))
            self._update_status(100, "Hoàn thành!")

        except Exception as e:
            self._update_status(0, f"Lỗi: {str(e)}")
            messagebox.showerror("Lỗi", f"Không thể phân tích:\n{str(e)}")

        self.window.after(0, lambda: self.btn_analyze.config(state=tk.NORMAL))

    def _display_result(self, final_score, s1, s2, s3, level, matches, weight_mode):
        total_suspect_sents = sum(m.get("num_sentences", 1) for m in matches)

        # Lý do paraphrase dựa trên chỉ số cao nhất
        reasons = []
        if s3 >= 0.80:
            reasons.append("mức độ trùng lặp ngữ nghĩa cao")
        if s1 >= 0.80:
            reasons.append("mức độ trùng lặp từ khóa cốt lõi cao")
        if s2 >= 0.80:
            reasons.append("mức độ sao chép nguyên văn cao")
        reason_str = f"(Dựa trên {reasons[0]})" if reasons else ""

        # Actionable insight
        insight = ""
        if s1 > 0.80 and s3 > 0.80:
            insight = (
                "👉 Lời khuyên: Đoạn văn có dấu hiệu xào nấu ý tưởng quá sát với nguồn gốc.\n"
                "   Hãy diễn đạt lại hoàn toàn bằng vốn từ của bạn hoặc thêm trích dẫn hợp lệ."
            )
        elif s2 > 0.80:
            insight = (
                "👉 Lời khuyên: Phát hiện sao chép nguyên văn đáng kể.\n"
                "   Cần đặt dấu ngoặc kép và trích dẫn nguồn ngay lập tức."
            )
        elif s1 > 0.80:
            insight = (
                "👉 Lời khuyên: Có trùng lặp từ khóa chính với nguồn tham khảo.\n"
                "   Hãy viết lại bằng cấu trúc câu và từ ngữ của riêng bạn."
            )
        elif s3 > 0.80:
            insight = (
                "👉 Lời khuyên: Ý tưởng và cách triển khai gần giống nguồn tham khảo.\n"
                "   Cần bổ sung phân tích, góc nhìn cá nhân hoặc trích dẫn rõ ràng."
            )

        text = (
            f"KẾT QUẢ PHÂN TÍCH ĐẠO VĂN\n"
            f"{'─'*50}\n"
            f"Mức độ cảnh báo: {level}\n"
            f"Tỷ lệ trùng lặp tổng thể: {final_score*100:.1f}%\n"
            f" {reason_str}\n"
            f"{'─'*50}\n"
            f"Chỉ số chi tiết:\n"
            f"  • Sao chép nguyên văn (Từng chữ):   {s2*100:5.1f}% ({_score_label(s2)})\n"
            f"  • Trùng lặp từ khóa cốt lõi:        {s1*100:5.1f}% ({_score_label(s1)})\n"
            f"  • Tương đồng về ý nghĩa (Ngữ nghĩa): {s3*100:5.1f}% ({_score_label(s3)})\n"
            f"{'─'*50}\n"
            f"Thống kê:\n"
            f"  • Số đoạn trùng phát hiện: {len(matches)} đoạn\n"
            f"  • Tổng số câu bị nghi ngờ: {total_suspect_sents} câu\n"
            f"{'─'*50}\n"
            f"{insight}"
        )

        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(1.0, text)
        self.result_text.config(state=tk.DISABLED)

        # Detail
        self.detail_text.config(state=tk.NORMAL)
        self.detail_text.delete(1.0, tk.END)
        detail = (
            f"Thông số thuật toán gốc:\n"
            f"  • Shingling (so khớp từng chữ): {s2*100:.1f}%\n"
            f"  • TF-IDF (tần số từ khóa):       {s1*100:.1f}%\n"
            f"  • BERT (ngữ nghĩa sâu):           {s3*100:.1f}%\n"
            f"{'─'*50}\n"
            f"Công thức trọng số:\n"
            f"  {weight_mode}\n"
            f"  = {final_score*100:.1f}%"
        )
        self.detail_text.insert(1.0, detail)
        self.detail_text.config(state=tk.DISABLED)

        # Ẩn detail frame mỗi khi có kết quả mới
        if self.detail_visible:
            self._toggle_detail()

    def _update_status(self, value, text):
        self.window.after(0, lambda: self.progress.configure(value=value))
        self.window.after(0, lambda: self.status_label.config(text=text))

    def _toggle_detail(self):
        if self.detail_visible:
            self.detail_frame.pack_forget()
            self.detail_btn.config(text="[ Xem chi tiết thuật toán (TF-IDF, Shingling, BERT) ]")
        else:
            self.detail_frame.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
            self.detail_btn.config(text="[ Ẩn chi tiết thuật toán ]")
        self.detail_visible = not self.detail_visible

    def _export_pdf(self):
        if not self.result_data:
            messagebox.showwarning("Cảnh báo", "Chưa có kết quả để xuất!")
            return
        os.makedirs(REPORTS_DIR, exist_ok=True)
        fname_a = os.path.splitext(os.path.basename(self.file_a_path))[0]
        fname_b = os.path.splitext(os.path.basename(self.file_b_path))[0]
        output = os.path.join(REPORTS_DIR, f"report_{fname_a}_vs_{fname_b}.pdf")
        try:
            generate_pairwise_report(
                os.path.basename(self.file_a_path),
                os.path.basename(self.file_b_path),
                self.result_data,
                output
            )
            messagebox.showinfo("Thành công", f"Đã xuất PDF:\n{output}")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Xuất PDF thất bại:\n{str(e)}")

    def _view_highlight(self):
        if not self.result_data:
            messagebox.showwarning("Cảnh báo", "Chưa có kết quả để hiển thị!")
            return
        try:
            HighlightViewerWindow(
                self.window,
                self.result_data["file_a_text"],
                self.result_data["file_b_text"],
                self.result_data["matches"],
                os.path.basename(self.file_a_path),
                os.path.basename(self.file_b_path),
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Lỗi Highlight", f"Không thể mở highlight:\n{e}")
