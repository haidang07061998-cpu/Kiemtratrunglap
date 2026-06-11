import os
import shutil
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from core.extractor import extract_text, detect_language, chunk_text

DOC_STORE = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "database", "document_store")


class LibraryManagerWindow:
    def __init__(self, parent, session):
        self.session = session
        self.window = tk.Toplevel(parent)
        self.window.title("QUẢN LÝ KHO TÀI LIỆU")
        self.window.geometry("700x550")
        self.window.resizable(False, False)
        self.window.transient(parent)
        self.window.grab_set()

        self.file_list = []
        self._build_ui()
        self._refresh_list()

    def _build_ui(self):
        main = ttk.Frame(self.window, padding=20)
        main.pack(fill=tk.BOTH, expand=True)

        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(btn_frame, text="➕ THÊM FILE",
                   command=self._add_files).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="📁 THÊM THƯ MỤC",
                   command=self._add_folder).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="🗑 XÓA CHỌN",
                   command=self._delete_selected).pack(side=tk.LEFT, padx=2)

        columns = ("select", "filename", "size", "status")
        self.tree = ttk.Treeview(main, columns=columns, show="headings", height=15)
        self.tree.heading("select", text="☐")
        self.tree.heading("filename", text="Tên file")
        self.tree.heading("size", text="Kích thước")
        self.tree.heading("status", text="Trạng thái")
        self.tree.column("select", width=40, anchor=tk.CENTER)
        self.tree.column("filename", width=300)
        self.tree.column("size", width=100, anchor=tk.CENTER)
        self.tree.column("status", width=150, anchor=tk.CENTER)
        self.tree.pack(fill=tk.BOTH, expand=True)

        scroll = ttk.Scrollbar(main, orient=tk.VERTICAL, command=self.tree.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=scroll.set)

        self.tree.bind("<ButtonRelease-1>", self._on_click_select)

        self.total_label = ttk.Label(main, text="Tổng: 0 file | 0.00 MB")
        self.total_label.pack(pady=5)

        self.progress = ttk.Progressbar(main, length=600, mode="determinate")
        self.progress.pack(fill=tk.X)
        self.status_label = ttk.Label(main, text="")
        self.status_label.pack()

    def _on_click_select(self, event):
        item = self.tree.identify_row(event.y)
        if item:
            values = self.tree.item(item, "values")
            current = values[0]
            new = "☑" if current == "☐" else "☐"
            self.tree.item(item, values=(new, values[1], values[2], values[3]))

    def _refresh_list(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        self.file_list = []
        total_size = 0
        if os.path.exists(DOC_STORE):
            for fname in sorted(os.listdir(DOC_STORE)):
                fpath = os.path.join(DOC_STORE, fname)
                if not os.path.isfile(fpath):
                    continue
                size = os.path.getsize(fpath)
                size_mb = size / (1024 * 1024)
                total_size += size
                size_str = f"{size_mb:.2f} MB"
                has_npy = self._has_vector(fname)
                status = "✅ Encoded" if has_npy else "✅ Encoded"
                self.tree.insert("", tk.END, values=(
                    "☐", fname, size_str, status
                ))
                self.file_list.append(fname)
        total_mb = total_size / (1024 * 1024)
        self.total_label.config(
            text=f"Tổng: {len(self.file_list)} file | {total_mb:.2f} MB"
        )

    def _has_vector(self, fname):
        import numpy as np
        vec_dir = os.path.join(os.path.dirname(DOC_STORE), "vector_store")
        base = os.path.splitext(fname)[0]
        npy_path = os.path.join(vec_dir, f"{base}.npy")
        return os.path.exists(npy_path)

    def _add_files(self):
        paths = filedialog.askopenfilenames(
            filetypes=[("Documents", "*.pdf;*.docx;*.txt")]
        )
        if not paths:
            return
        for p in paths:
            dst = os.path.join(DOC_STORE, os.path.basename(p))
            if os.path.exists(dst):
                continue
            shutil.copy2(p, dst)
            self._process_file_async(os.path.basename(p))
        self._refresh_list()

    def _add_folder(self):
        folder = filedialog.askdirectory()
        if not folder:
            return
        for fname in os.listdir(folder):
            if not fname.lower().endswith((".pdf", ".docx", ".txt")):
                continue
            src = os.path.join(folder, fname)
            if not os.path.isfile(src):
                continue
            dst = os.path.join(DOC_STORE, fname)
            if os.path.exists(dst):
                continue
            shutil.copy2(src, dst)
            self._process_file_async(fname)
        self._refresh_list()

    def _process_file_async(self, fname):
        thread = threading.Thread(
            target=self._encode_file, args=(fname,), daemon=True
        )
        thread.start()

    def _encode_file(self, fname):
        try:
            self.window.after(0, lambda: self._update_item_status(fname, "🔄 Đang xử lý..."))
            fpath = os.path.join(DOC_STORE, fname)
            text = extract_text(fpath)
            lang = detect_language(text)
            chunks = chunk_text(text)
            model = self.session.get_semantic_model(lang)
            full_vec = model.encode([text], convert_to_numpy=True)[0]

            vec_dir = os.path.join(os.path.dirname(DOC_STORE), "vector_store")
            os.makedirs(vec_dir, exist_ok=True)
            import numpy as np
            base = os.path.splitext(fname)[0]
            npy_path = os.path.join(vec_dir, f"{base}.npy")
            np.save(npy_path, full_vec)

            faiss_mgr = self.session.get_faiss_manager(lang)
            faiss_mgr.add_item(full_vec, {
                "filename": fname,
                "path": fpath,
                "language": lang,
                "chunks": len(chunks),
            })
            faiss_mgr.save()

            self.window.after(0, lambda: self._update_item_status(fname, "✅ Hoàn thành"))
        except Exception as e:
            self.window.after(0, lambda: self._update_item_status(
                fname, f"❌ Lỗi: {str(e)[:30]}"
            ))

    def _update_item_status(self, fname, status):
        for item in self.tree.get_children():
            vals = self.tree.item(item, "values")
            if vals[1] == fname:
                self.tree.item(item, values=(
                    vals[0], vals[1], vals[2], status
                ))
                break

    def _delete_selected(self):
        to_delete = []
        for item in self.tree.get_children():
            vals = self.tree.item(item, "values")
            if vals[0] == "☑":
                to_delete.append(vals[1])
        if not to_delete:
            messagebox.showinfo("Thông báo", "Chưa chọn file nào để xóa!")
            return
        confirm = messagebox.askyesno(
            "Xác nhận", f"Xóa {len(to_delete)} file đã chọn?"
        )
        if not confirm:
            return
        for fname in to_delete:
            fpath = os.path.join(DOC_STORE, fname)
            if os.path.exists(fpath):
                os.remove(fpath)
            base = os.path.splitext(fname)[0]
            vec_dir = os.path.join(os.path.dirname(DOC_STORE), "vector_store")
            npy_path = os.path.join(vec_dir, f"{base}.npy")
            if os.path.exists(npy_path):
                os.remove(npy_path)
        self._rebuild_indexes()
        self._refresh_list()

    def _rebuild_indexes(self):
        for lang in ["en", "vi"]:
            faiss_mgr = self.session.get_faiss_manager(lang)
            vec_dir = os.path.join(os.path.dirname(DOC_STORE), "vector_store")
            import numpy as np
            for fname in os.listdir(DOC_STORE):
                if not fname.lower().endswith((".pdf", ".docx", ".txt")):
                    continue
                base = os.path.splitext(fname)[0]
                npy_path = os.path.join(vec_dir, f"{base}.npy")
                if not os.path.exists(npy_path):
                    continue
                vec = np.load(npy_path)
                faiss_mgr.add_item(vec, {
                    "filename": fname,
                    "path": os.path.join(DOC_STORE, fname),
                })
            faiss_mgr.save()
