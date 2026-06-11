import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from gui.session import AppSession
from core.trainer import train_ensemble


def print_progress(value, text):
    bar_len = 40
    filled = int(bar_len * value / 100)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\r{bar} {value:3d}% | {text}", end="", flush=True)
    if value == 100:
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=["en", "vi"], default="en",
                        help="Ngôn ngữ: en (PAN-PC-11) hoặc vi (ViSP)")
    parser.add_argument("--mode", choices=["stage1", "stage2"], default="stage1",
                        help="stage1=English PAN-PC-11, stage2=Vietnamese ViSP")
    args = parser.parse_args()

    lang = args.lang
    if args.mode == "stage2":
        lang = "vi"

    stage_name = "GIAI ĐOẠN 1: PAN-PC-11 (Tiếng Anh)" if lang == "en" else "GIAI ĐOẠN 2: ViSP (Tiếng Việt)"
    print("=" * 60)
    print(f"  {stage_name}")
    print("=" * 60)
    print()

    print("[1/3] Đang tải mô hình nền...")
    session = AppSession()
    session.load_semantic_models(progress_callback=print_progress)
    print()

    print("[2/3] Đang huấn luyện...")
    start = time.time()
    model = train_ensemble(session, progress_callback=print_progress, language=lang)
    elapsed = time.time() - start
    print()

    fname = "ensemble_model_pan.pkl" if lang == "en" else "ensemble_model_vi.pkl"
    print(f"[3/3] Hoàn thành trong {elapsed:.1f}s")
    print(f"  • Accuracy: {model['accuracy']*100:.2f}%")
    print(f"  • Model saved: models/{fname}")
    print()
    print("=" * 60)
    print("  CHẠY: python main.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
