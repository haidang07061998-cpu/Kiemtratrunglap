import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from gui.session import AppSession
from core.extractor import extract_text, detect_language, chunk_text

DOC_STORE = os.path.join(os.path.dirname(__file__), "database", "document_store")
VEC_STORE = os.path.join(os.path.dirname(__file__), "database", "vector_store")


def main():
    print("=" * 60)
    print("  BUILD VECTOR EMBEDDINGS - KHO TÀI LIỆU")
    print("=" * 60)
    print()

    print("[1/3] Đang tải mô hình AI...")
    session = AppSession()
    session.load_semantic_models(progress_callback=lambda v, t: print(f"  {v:3d}% | {t}"))
    print()

    pdf_files = sorted([
        f for f in os.listdir(DOC_STORE)
        if f.lower().endswith(".pdf") and os.path.isfile(os.path.join(DOC_STORE, f))
    ])
    total = len(pdf_files)
    print(f"[2/3] Tìm thấy {total} file PDF trong document_store/")
    print()

    success = 0
    failed = 0
    start_time = time.time()

    for i, fname in enumerate(pdf_files, 1):
        fpath = os.path.join(DOC_STORE, fname)
        base = os.path.splitext(fname)[0]
        npy_path = os.path.join(VEC_STORE, f"{base}.npy")

        # Skip if already encoded
        if os.path.exists(npy_path):
            print(f"  [{i}/{total}] ⏭ {fname} (đã encode)")
            success += 1
            continue

        try:
            print(f"  [{i}/{total}] Đang xử lý {fname}...", end=" ")

            text = extract_text(fpath)
            if not text.strip():
                print("❌ Text rỗng")
                failed += 1
                continue

            lang = detect_language(text)
            chunks = chunk_text(text)
            model = session.get_semantic_model(lang)

            full_vec = model.encode([text], convert_to_numpy=True)[0]

            os.makedirs(VEC_STORE, exist_ok=True)
            import numpy as np
            np.save(npy_path, full_vec)

            faiss_mgr = session.get_faiss_manager(lang)
            faiss_mgr.add_item(full_vec, {
                "filename": fname,
                "path": fpath,
                "language": lang,
                "chunks": len(chunks),
            })
            faiss_mgr.save()

            print(f"✅ ({lang}, {len(chunks)} chunks)")
            success += 1

        except Exception as e:
            print(f"❌ Lỗi: {str(e)[:60]}")
            failed += 1

    elapsed = time.time() - start_time
    print()
    print("[3/3] Kết quả:")
    print(f"  • Thành công: {success}/{total}")
    print(f"  • Thất bại:   {failed}/{total}")
    print(f"  • Thời gian:   {elapsed:.1f}s")
    print()

    # Show FAISS index stats
    for lang in ["en", "vi"]:
        mgr = session.get_faiss_manager(lang)
        count = mgr.get_total_count()
        if count > 0:
            print(f"  FAISS index '{lang}': {count} vectors")
    print()
    print("=" * 60)
    print("  HOÀN TẤT! Chạy 'python main.py' để mở GUI.")
    print("=" * 60)


if __name__ == "__main__":
    main()
