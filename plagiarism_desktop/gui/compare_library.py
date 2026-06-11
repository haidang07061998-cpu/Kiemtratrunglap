import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from core.extractor import extract_text, detect_language, chunk_text
from core.preprocessor import preprocess
from core.similarity import compute_ensemble_score, classify_plagiarism
from core.report_generator import generate_library_report

REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
DOC_STORE = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "database", "document_store")
VECTOR_STORE = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "database", "vector_store")


class CompareLibraryWindow:
    def __init__(self, parent, session):
        self.session = session
        self.window = tk.Toplevel(parent)
        self.window.title("SO SÁNH VỚI KHO")
        self.window.geometry("700x600")
        self.window.minsize(600, 400)

        self.file_path = None
        self.result_data = None

        self._build_ui()

    def _build_ui(self):
        main = ttk.Frame(self.window, padding=15)
        main.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main, text="FILE CẦN KIỂM TRA:").pack(anchor=tk.W)
        file_sel = ttk.Frame(main)
        file_sel.pack(fill=tk.X, pady=5)
        self.entry_file = ttk.Entry(file_sel)
        self.entry_file.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(file_sel, text="CHỌN", command=self._choose_file).pack(side=tk.LEFT)

        lang_frame = ttk.Frame(main)
        lang_frame.pack(fill=tk.X, pady=5)
        ttk.Label(lang_frame, text="NGÔN NGỮ:").pack(side=tk.LEFT, padx=(0, 10))
        self.lang_var = tk.StringVar(self.window, value="auto")
        self.lang_combo = ttk.Combobox(
            lang_frame, textvariable=self.lang_var,
            values=["auto", "en", "vi"], state="readonly", width=15
        )
        self.lang_combo.pack(side=tk.LEFT)

        self.btn_analyze = ttk.Button(
            main, text="▶ BẮT ĐẦU PHÂN TÍCH", command=self._start_analysis
        )
        self.btn_analyze.pack(pady=10)

        self.progress = ttk.Progressbar(main, length=600, mode="determinate")
        self.progress.pack(fill=tk.X, pady=5)
        self.status_label = ttk.Label(main, text="")
        self.status_label.pack()

        result_frame = ttk.LabelFrame(main, text="KẾT QUẢ", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.result_text = tk.Text(result_frame, height=12, state=tk.DISABLED)
        scroll = ttk.Scrollbar(result_frame, orient=tk.VERTICAL,
                               command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scroll.set)
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Button(main, text="📑 XUẤT PDF",
                   command=self._export_pdf).pack(pady=5)

    def _choose_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("Documents", "*.pdf;*.docx;*.txt")]
        )
        if path:
            self.file_path = path
            self.entry_file.delete(0, tk.END)
            self.entry_file.insert(0, os.path.basename(path))

    def _start_analysis(self):
        if not self.file_path:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn file cần kiểm tra!")
            return

        lib_files = [f for f in os.listdir(DOC_STORE)
                     if f.lower().endswith((".pdf", ".docx", ".txt"))]
        if not lib_files:
            messagebox.showwarning("Cảnh báo", "Kho tài liệu trống! Hãy thêm file vào Document Library.")
            return

        self.btn_analyze.config(state=tk.DISABLED)
        self.progress["value"] = 0
        self.status_label.config(text="Đang phân tích...")

        thread = threading.Thread(target=self._run_analysis, args=(lib_files,), daemon=True)
        thread.start()

    def _run_analysis(self, lib_files):
        try:
            lang = self.lang_var.get()
            if lang == "auto":
                self._update_status(5, "Đang nhận diện ngôn ngữ...")
                query_text = extract_text(self.file_path)
                lang = detect_language(query_text)
            else:
                query_text = extract_text(self.file_path)

            self._update_status(10, "Đang encode vector...")
            faiss_mgr = self.session.get_faiss_manager(lang)
            model = self.session.get_semantic_model(lang)
            query_chunks = chunk_text(query_text)
            query_vec = model.encode([query_text], convert_to_numpy=True)[0]

            self._update_status(20, "Đang tìm kiếm FAISS...")
            top_results = faiss_mgr.search(query_vec, top_k=20)

            self._update_status(30, "Đang phân tích chi tiết...")
            detailed = []
            total = len(top_results)
            for i, r in enumerate(top_results):
                meta = r["metadata"]
                src_path = meta.get("path", "")
                if not os.path.exists(src_path):
                    continue
                src_text = extract_text(src_path)
                fs, s1, s2, s3 = compute_ensemble_score(
                    query_text, src_text, lang, self.session
                )
                detailed.append({
                    "filename": meta.get("filename", os.path.basename(src_path)),
                    "path": src_path,
                    "score": fs,
                    "s1": s1,
                    "s2": s2,
                    "s3": s3,
                    "level": classify_plagiarism(fs, s1, s2, s3),
                })
                self._update_status(
                    30 + int((i+1)/total*50),
                    f"Đã phân tích {i+1}/{total} nguồn..."
                )

            detailed.sort(key=lambda x: x["score"], reverse=True)
            top5 = detailed[:5]

            avg_score = sum(d["score"] for d in detailed) / len(detailed) if detailed else 0
            level = classify_plagiarism(avg_score)

            self.result_data = {
                "avg_score": avg_score,
                "level": level,
                "top_sources": top5,
                "all_sources": detailed,
                "filename": os.path.basename(self.file_path),
            }

            lines = [f"Độ trùng lặp TB: {avg_score*100:.1f}%",
                     f"Mức đánh giá: {level}\n",
                     "📄 TOP NGUỒN TRÙNG:"]
            for i, src in enumerate(top5, 1):
                lines.append(
                    f"  {i}. {src['filename']} - {src['score']*100:.1f}% {src['level'][:1]}"
                )
            self._update_result("\n".join(lines), 100)

        except Exception as e:
            self._update_status(0, f"Lỗi: {str(e)}")
            messagebox.showerror("Lỗi", str(e))

        self.window.after(0, lambda: self.btn_analyze.config(state=tk.NORMAL))

    def _update_status(self, value, text):
        self.window.after(0, lambda: self.progress.configure(value=value))
        self.window.after(0, lambda: self.status_label.config(text=text))

    def _update_result(self, text, progress):
        self.window.after(0, lambda: self.progress.configure(value=progress))
        self.window.after(0, lambda: self.status_label.config(text="Hoàn thành!"))
        self.window.after(0, lambda: self._set_result_text(text))

    def _set_result_text(self, text):
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(1.0, text)
        self.result_text.config(state=tk.DISABLED)

    def _export_pdf(self):
        if not self.result_data:
            messagebox.showwarning("Cảnh báo", "Chưa có kết quả để xuất!")
            return
        os.makedirs(REPORTS_DIR, exist_ok=True)
        fname = os.path.splitext(os.path.basename(self.file_path))[0]
        output = os.path.join(REPORTS_DIR, f"report_{fname}_vs_library.pdf")
        try:
            generate_library_report(
                os.path.basename(self.file_path),
                self.result_data,
                output
            )
            messagebox.showinfo("Thành công", f"Đã xuất PDF:\n{output}")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Xuất PDF thất bại:\n{str(e)}")
