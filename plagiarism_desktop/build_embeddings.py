import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(__file__))

from gui.session import AppSession
from core.extractor import extract_text, detect_language, chunk_text

DOC_STORE = os.path.join(os.path.dirname(__file__), "database", "document_store")
VEC_STORE = os.path.join(os.path.dirname(__file__), "database", "vector_store")


def main():
    print("=" * 60)
    print("  BUILD VECTOR EMBEDDINGS - KHO TÀI LIỆU (CHUNK-LEVEL)")
    print("=" * 60)
    print()

    print("[1/3] Đang tải mô hình AI...")
    session = AppSession()
    session.load_semantic_models(progress_callback=lambda v, t: print(f"  {v:3d}% | {t}"))
    print()

    # Support PDF, DOCX, TXT
    ALLOWED_EXT = (".pdf", ".docx", ".txt")
    all_files = sorted([
        f for f in os.listdir(DOC_STORE)
        if f.lower().endswith(ALLOWED_EXT) and os.path.isfile(os.path.join(DOC_STORE, f))
    ])
    total = len(all_files)
    print(f"[2/3] Tìm thấy {total} file trong document_store/")
    print()

    # Reset FAISS indexes for clean rebuild
    from core.faiss_manager import FAISSManager
    for lang in ["en", "vi"]:
        mgr = FAISSManager(lang)
        mgr.reset()

    faiss_mgrs = {"en": FAISSManager("en"), "vi": FAISSManager("vi")}

    success = 0
    failed = 0
    start_time = time.time()

    for i, fname in enumerate(all_files, 1):
        fpath = os.path.join(DOC_STORE, fname)
        base = os.path.splitext(fname)[0]

        try:
            print(f"  [{i}/{total}] Đang xử lý {fname}...", end=" ")

            text = extract_text(fpath)
            if not text.strip():
                print("❌ Text rỗng")
                failed += 1
                continue

            lang = detect_language(text)
            model = session.get_semantic_model(lang)

            # FIX: Encode theo chunk
            chunks = chunk_text(text, max_words=256, overlap=50)
            if not chunks:
                print("❌ Không có chunk")
                failed += 1
                continue

            chunk_vecs = model.encode(chunks, convert_to_numpy=True)

            os.makedirs(VEC_STORE, exist_ok=True)
            for ci, (chunk_vec, chunk_text_val) in enumerate(zip(chunk_vecs, chunks)):
                faiss_mgrs[lang].add_item(chunk_vec, {
                    "filename": fname,
                    "path": fpath,
                    "language": lang,
                    "chunk_idx": ci,
                    "num_chunks": len(chunks),
                })

            # FIX: Lưu meta.json thay vì .npy (không cần 1-vector/file nữa)
            meta_path = os.path.join(VEC_STORE, f"{base}.meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({"filename": fname, "language": lang, "num_chunks": len(chunks)}, f)

            print(f"✅ ({lang}, {len(chunks)} chunks)")
            success += 1

        except Exception as e:
            print(f"❌ Lỗi: {str(e)[:60]}")
            import traceback; traceback.print_exc()
            failed += 1

    for lang, mgr in faiss_mgrs.items():
        mgr.save()

    elapsed = time.time() - start_time
    print()
    print("[3/3] Kết quả:")
    print(f"  • Thành công: {success}/{total}")
    print(f"  • Thất bại:   {failed}/{total}")
    print(f"  • Thời gian:   {elapsed:.1f}s")
    print()

    for lang in ["en", "vi"]:
        count = faiss_mgrs[lang].get_total_count()
        if count > 0:
            print(f"  FAISS index '{lang}': {count} vectors")
    print()
    print("=" * 60)
    print("  HOÀN TẤT! Chạy 'python main.py' để mở GUI.")
    print("=" * 60)


if __name__ == "__main__":
    main()
