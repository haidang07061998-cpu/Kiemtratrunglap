import os
import sys
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw

sys.path.insert(0, os.path.dirname(__file__))

from gui.session import AppSession
from gui.main_window import MainWindow


class SplashScreen:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("PLAGIARISM DETECTOR AI - Đang tải...")
        self.root.attributes("-topmost", True)

        w, h = 500, 300
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x = (sw - w) // 2
        y = (sh - h) // 2
        self.root.geometry(f"{w}x{h}+{x}+{y}")

        self.root.configure(bg="#2c3e50")

        frame = tk.Frame(self.root, bg="#2c3e50")
        frame.pack(expand=True, fill=tk.BOTH, padx=30, pady=20)

        tk.Label(
            frame, text="PLAGIARISM DETECTOR AI",
            font=("Arial", 18, "bold"),
            fg="white", bg="#2c3e50"
        ).pack(pady=(30, 10))

        self.progress = ttk.Progressbar(
            frame, length=350, mode="determinate"
        )
        self.progress.pack(pady=20)

        self.status = tk.StringVar(value="Đang tải các mô hình AI, vui lòng chờ giây lát...")
        tk.Label(
            frame, textvariable=self.status,
            font=("Arial", 10), fg="#bdc3c7", bg="#2c3e50",
            wraplength=400
        ).pack(pady=10)

        self.root.update()

    def update_progress(self, value, text):
        self.progress["value"] = value
        self.status.set(text)
        self.root.update()

    def close(self):
        self.root.destroy()


def main():
    splash = SplashScreen()

    def load_callback(value, text):
        print(f"[{value:3d}%] {text}")
        splash.root.after(0, lambda: splash.update_progress(value, text))

    def load_models():
        session = AppSession()
        session.load_semantic_models(progress_callback=load_callback)
        splash.root.after(0, lambda: _start_gui(session))

    def _start_gui(session):
        app = MainWindow(session)
        splash.close()
        app.window.mainloop()

    thread = threading.Thread(target=load_models, daemon=True)
    thread.start()

    splash.root.mainloop()


if __name__ == "__main__":
    main()
