import os
import tkinter as tk
from tkinter import ttk, messagebox

from gui.compare_two import CompareTwoWindow
from gui.compare_library import CompareLibraryWindow
from gui.library_manager import LibraryManagerWindow


class MainWindow:
    def __init__(self, session):
        self.session = session
        self.window = tk.Tk()
        self.window.title("PLAGIARISM DETECTOR AI")
        self.window.geometry("600x500")
        self.window.resizable(False, False)

        self._build_ui()

    def _build_ui(self):
        main_frame = ttk.Frame(self.window, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(
            main_frame,
            text="PLAGIARISM DETECTOR AI",
            font=("Arial", 18, "bold")
        )
        title.pack(pady=(0, 20))

        stats_frame = ttk.LabelFrame(main_frame, text="THỐNG KÊ KHO", padding=10)
        stats_frame.pack(fill=tk.X, pady=(0, 20))

        doc_store = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "database", "document_store"
        )
        total_files = 0
        total_size = 0
        if os.path.exists(doc_store):
            for f in os.listdir(doc_store):
                fpath = os.path.join(doc_store, f)
                if os.path.isfile(fpath):
                    total_files += 1
                    total_size += os.path.getsize(fpath)

        size_mb = total_size / (1024 * 1024)
        self.stats_label = ttk.Label(
            stats_frame,
            text=f"Số file: {total_files} file  |  Dung lượng: {size_mb:.2f} MB"
        )
        self.stats_label.pack()

        btn_frame = ttk.LabelFrame(main_frame, text="CHỌN CHẾ ĐỘ LÀM VIỆC", padding=20)
        btn_frame.pack(fill=tk.BOTH, expand=True)

        btn_compare_two = ttk.Button(
            btn_frame,
            text="📄 SO SÁNH 2 FILE",
            command=self._open_compare_two,
            width=30
        )
        btn_compare_two.pack(pady=8)

        btn_compare_lib = ttk.Button(
            btn_frame,
            text="📚 SO SÁNH VỚI KHO",
            command=self._open_compare_library,
            width=30
        )
        btn_compare_lib.pack(pady=8)

        btn_lib_mgr = ttk.Button(
            btn_frame,
            text="⚙ QUẢN LÝ KHO TÀI LIỆU",
            command=self._open_library_manager,
            width=30
        )
        btn_lib_mgr.pack(pady=8)

        btn_exit = ttk.Button(
            main_frame,
            text="🚪 THOÁT",
            command=self.window.quit,
            width=20
        )
        btn_exit.pack(pady=(20, 0))

    def _open_compare_two(self):
        import traceback
        try:
            CompareTwoWindow(self.window, self.session)
        except Exception as e:
            traceback.print_exc()
            from tkinter import messagebox
            messagebox.showerror("Lỗi", f"Không thể mở cửa sổ so sánh:\n{str(e)}")

    def _open_compare_library(self):
        CompareLibraryWindow(self.window, self.session)

    def _open_library_manager(self):
        LibraryManagerWindow(self.window, self.session)

    def refresh_stats(self):
        doc_store = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "database", "document_store"
        )
        total_files = 0
        total_size = 0
        if os.path.exists(doc_store):
            for f in os.listdir(doc_store):
                fpath = os.path.join(doc_store, f)
                if os.path.isfile(fpath):
                    total_files += 1
                    total_size += os.path.getsize(fpath)
        size_mb = total_size / (1024 * 1024)
        self.stats_label.config(
            text=f"Số file: {total_files} file  |  Dung lượng: {size_mb:.2f} MB"
        )
