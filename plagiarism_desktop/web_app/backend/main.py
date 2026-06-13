import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
import sys
import asyncio
import threading
import uuid
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from core.extractor import extract_text, detect_language, chunk_text, count_words
from core.similarity import compute_ensemble_score, classify_plagiarism, compute_tfidf_similarity, compute_shingling_similarity
from core.highlighter import find_matching_positions, split_sentences, compute_coverage
from core.report_generator import generate_pairwise_report, generate_library_report
from core.faiss_manager import FAISSManager
from core.library_cache import preprocess_and_cache, load_text, clear_file_cache
from gui.session import AppSession

# FIX: helper sanitize filename
def _safe_name(name):
    return Path(name).name

app = FastAPI(title="Plagiarism Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=2)
session_lock = threading.Lock()
_session = None
_models_ready = threading.Event()

DOC_STORE = Path(__file__).resolve().parents[2] / "database" / "document_store"
VEC_STORE = Path(__file__).resolve().parents[2] / "database" / "vector_store"
REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"
DOC_STORE.mkdir(parents=True, exist_ok=True)
VEC_STORE.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


def _load_models_blocking():
    global _session
    s = AppSession()
    s.load_semantic_models()
    _session = s
    _models_ready.set()
    print("[STARTUP] Models loaded successfully")


@app.on_event("startup")
async def startup():
    loop = asyncio.get_event_loop()
    print("[STARTUP] Loading AI models, please wait...")
    await loop.run_in_executor(executor, _load_models_blocking)
    print("[STARTUP] Server ready")


def get_session():
    return _session


@app.get("/api/stats")
async def get_stats():
    total_files = 0
    total_size = 0
    if DOC_STORE.exists():
        for f in DOC_STORE.iterdir():
            if f.is_file():
                total_files += 1
                total_size += f.stat().st_size
    size_mb = total_size / (1024 * 1024)
    return {"total_files": total_files, "total_size_mb": round(size_mb, 2)}


@app.get("/api/library/files")
async def list_library_files():
    files = []
    if DOC_STORE.exists():
        for f in sorted(DOC_STORE.iterdir()):
            if f.suffix.lower() not in (".pdf", ".docx", ".txt"):
                continue
            size_mb = f.stat().st_size / (1024 * 1024)
            base = f.stem
            # FIX: dùng meta.json thay vì .npy
            meta_path = VEC_STORE / f"{base}.meta.json"
            encoded = meta_path.exists()
            lang = "en"
            if encoded:
                try:
                    with open(meta_path, "r", encoding="utf-8") as mf:
                        meta = json.load(mf)
                        lang = meta.get("language", "en")
                except Exception:
                    pass
            else:
                lang = _detect_file_lang(f)
            files.append({
                "name": f.name,
                "size_mb": round(size_mb, 2),
                "encoded": encoded,
                "language": lang,
            })
    return {"files": files}


def _detect_file_lang(fpath):
    try:
        text = extract_text(str(fpath))
        return detect_language(text)
    except Exception:
        return "en"


@app.post("/api/library/upload")
async def upload_to_library(files: list[UploadFile] = File(...)):
    saved = []
    for f in files:
        safe_name = _safe_name(f.filename)
        dst = DOC_STORE / safe_name
        if dst.exists():
            continue
        content = await f.read()
        dst.write_bytes(content)
        saved.append(safe_name)
    if saved:
        _invalidate_text_cache()
    return {"saved": saved}


@app.post("/api/library/delete")
async def delete_library_files(data: dict):
    names = data.get("names", [])
    deleted = []
    for name in names:
        safe_name = _safe_name(name)
        fpath = DOC_STORE / safe_name
        if fpath.exists():
            fpath.unlink()
            deleted.append(safe_name)
        base = Path(safe_name).stem
        # FIX: xóa cả .npy, .meta.json và cache
        for ext in [".npy", ".meta.json"]:
            p = VEC_STORE / f"{base}{ext}"
            if p.exists():
                p.unlink()
        clear_file_cache(base)
    _rebuild_all_indexes()
    _invalidate_text_cache()
    return {"deleted": deleted}


def _rebuild_all_indexes():
    try:
        session = get_session()
        for lang in ["en", "vi"]:
            # FIX: dùng reset() thay vì xóa file thủ công
            faiss_mgr = FAISSManager(lang)
            faiss_mgr.reset()
            for fpath in DOC_STORE.iterdir():
                if fpath.suffix.lower() not in (".pdf", ".docx", ".txt"):
                    continue
                base = fpath.stem
                meta_path = VEC_STORE / f"{base}.meta.json"
                if not meta_path.exists():
                    continue
                with open(meta_path, "r", encoding="utf-8") as mf:
                    meta = json.load(mf)
                file_lang = meta.get("language", "en")
                if file_lang != lang:
                    continue
                # FIX: đọc chunk vectors từ meta
                num_chunks = meta.get("num_chunks", 0)
                if num_chunks == 0:
                    continue
                # Thử load .npy (backward compat), nếu không có thì rebuild
                npy_path = VEC_STORE / f"{base}.npy"
                if npy_path.exists():
                    vec = np.load(str(npy_path))
                    faiss_mgr.add_item(vec, {
                        "filename": fpath.name,
                        "path": str(fpath),
                        "language": lang,
                    })
                # Nếu không, cần encode lại (sẽ được xử lý bởi encode-all)
            faiss_mgr.save()
            session.faiss_managers[lang] = faiss_mgr
    except Exception as e:
        print(f"Rebuild index error: {e}")
        import traceback; traceback.print_exc()


@app.post("/api/library/encode-all")
async def encode_all_library_files():
    session = get_session()
    for fpath in DOC_STORE.iterdir():
        if fpath.suffix.lower() not in (".pdf", ".docx", ".txt"):
            continue
        base = fpath.stem
        meta_path = VEC_STORE / f"{base}.meta.json"
        if meta_path.exists():
            continue
        try:
            text = extract_text(str(fpath))
            if not text.strip():
                continue
            lang = detect_language(text)
            model = session.get_semantic_model(lang)

            # FIX: Encode theo chunk
            chunks = chunk_text(text, max_words=256, overlap=50)
            chunk_vecs = model.encode(chunks, convert_to_numpy=True)

            faiss_mgr = session.get_faiss_manager(lang)
            for ci, chunk_vec in enumerate(chunk_vecs):
                faiss_mgr.add_item(chunk_vec, {
                    "filename": fpath.name,
                    "path": str(fpath),
                    "language": lang,
                    "chunk_idx": ci,
                    "num_chunks": len(chunks),
                })
            faiss_mgr.save()

            # FIX: Lưu meta.json
            with open(meta_path, "w", encoding="utf-8") as mf:
                json.dump({"filename": fpath.name, "language": lang, "num_chunks": len(chunks)}, mf)

        except Exception as e:
            print(f"Encode error {fpath.name}: {e}")
            import traceback; traceback.print_exc()
        try:
            preprocess_and_cache(str(fpath), base, text, lang, session)
        except Exception as e:
            print(f"Cache error {fpath.name}: {e}")
    _invalidate_text_cache()
    return {"status": "done"}


class CompareTwoRequest(BaseModel):
    file_a_name: str = ""
    file_a_content: str = ""
    file_b_name: str = ""
    file_b_content: str = ""
    language: str = "auto"


@app.post("/api/compare-two")
async def compare_two_files(data: CompareTwoRequest):
    session = get_session()
    text_a = data.file_a_content
    text_b = data.file_b_content
    if not text_a or not text_b:
        raise HTTPException(400, "Missing file content")

    lang = data.language
    if lang == "auto":
        lang_a = detect_language(text_a)
        lang_b = detect_language(text_b)
        lang = lang_a if lang_a == lang_b else "en"

    topic_score, s1, s2, s3 = compute_ensemble_score(text_a, text_b, lang, session)
    matches = find_matching_positions(text_a, text_b, lang, session)
    coverage_a, coverage_b = compute_coverage(text_a, text_b, matches)
    level = classify_plagiarism(topic_score, s1, s2, s3)

    ensemble = session.get_ensemble_model(lang)
    if ensemble is not None and "classifier" in ensemble:
        weight_mode = "Kết hợp: Ensemble (LogisticRegression) + trọng số động"
    elif s3 > 0.9 or s1 > 0.9:
        weight_mode = "Paraphrase: Ngữ nghĩa×0.5 + Từ khóa×0.3 + Nguyên văn×0.2"
    else:
        weight_mode = "Mặc định: Nguyên văn×0.5 + Từ khóa×0.3 + Ngữ nghĩa×0.2"

    # FIX: plagiarism_score = % ký tự highlight / tổng ký tự (như Turnitin)
    total_chars = len(text_a.strip())
    if total_chars > 0:
        highlight_chars = 0
        for m in matches:
            if m.get("type") not in ("copy", "paraphrase"):
                continue
            spans = m.get("orig_spans")
            if spans:
                highlight_chars += sum(end - start for start, end in spans)
            else:
                highlight_chars += m.get("orig_end", 0) - m.get("orig_start", 0)
        plagiarism_score = highlight_chars / total_chars
    else:
        plagiarism_score = 0.0

    # FIX: Dùng split_sentences_with_offsets để đồng bộ index với find_matching_positions
    from core.highlighter import split_sentences_with_offsets
    sents_a = split_sentences_with_offsets(text_a, lang)
    sents_b = split_sentences_with_offsets(text_b, lang)
    sentences_a = [s["text"] for s in sents_a]
    sentences_b = [s["text"] for s in sents_b]

    matches_clean = []
    for m in matches:
        matches_clean.append({
            "orig_text": m["orig_text"],
            "susp_text": m["susp_text"],
            "score": round(m["score"], 4),
            "type": m["type"],
            "num_sentences": m.get("num_sentences", 1),
            "orig_idx_start": m.get("orig_idx_start", 0),
            "orig_idx_end": m.get("orig_idx_end", 0),
            "orig_indices": m.get("orig_indices", list(range(m["orig_idx_start"], m["orig_idx_end"] + 1))),
            "susp_idx_start": m.get("susp_idx_start", 0),
            "susp_idx_end": m.get("susp_idx_end", 0),
            "susp_indices": m.get("susp_indices", list(range(m["susp_idx_start"], m["susp_idx_end"] + 1))),
            "is_common_knowledge": m.get("is_common_knowledge", False) or m["type"] == "common_knowledge",
        })

    # Chia nhỏ copy vs paraphrase trong plagiarism_score
    def _chars(m):
        spans = m.get("orig_spans")
        if spans:
            return sum(end - start for start, end in spans)
        return m.get("orig_end", 0) - m.get("orig_start", 0)
    copy_chars = sum(_chars(m) for m in matches if m.get("type") == "copy")
    para_chars = sum(_chars(m) for m in matches if m.get("type") == "paraphrase")

    return {
        "plagiarism_score": round(plagiarism_score, 4),
        "copy_score": round(copy_chars / total_chars, 4) if total_chars > 0 else 0.0,
        "paraphrase_score": round(para_chars / total_chars, 4) if total_chars > 0 else 0.0,
        "topic_score": round(topic_score, 4),
        "score_1": round(s1, 4),
        "score_2": round(s2, 4),
        "score_3": round(s3, 4),
        "coverage_a": round(coverage_a, 4),
        "coverage_b": round(coverage_b, 4),
        "level": level,
        "matches": matches_clean,
        "sentences_a": sentences_a,
        "sentences_b": sentences_b,
        "text_a": text_a,
        "text_b": text_b,
        "language": lang,
        "weight_mode": weight_mode,
        "file_a_name": data.file_a_name,
        "file_b_name": data.file_b_name,
    }


_text_cache: dict[str, str] = {}
_text_cache_lock = threading.Lock()

def _get_or_extract_text(path: str) -> str:
    with _text_cache_lock:
        if path not in _text_cache:
            _text_cache[path] = extract_text(path)
        return _text_cache[path]

def _invalidate_text_cache(path: str = None):
    with _text_cache_lock:
        if path:
            _text_cache.pop(path, None)
        else:
            _text_cache.clear()

@app.post("/api/compare-library")
async def compare_with_library(data: dict):
    session = get_session()
    file_content = data.get("file_content", "")
    file_name = data.get("file_name", "")
    lang = data.get("language", "auto")

    if not file_content:
        raise HTTPException(400, "Missing file content")

    if lang == "auto":
        lang = detect_language(file_content)

    model = session.get_semantic_model(lang)
    query_chunks = chunk_text(file_content, max_words=256, overlap=50)

    loop = asyncio.get_event_loop()
    query_vecs = await loop.run_in_executor(
        executor, lambda: model.encode(query_chunks, convert_to_numpy=True)
    )

    faiss_mgr = session.get_faiss_manager(lang)
    total_vecs = faiss_mgr.get_total_count()
    file_scores: dict[str, dict] = {}
    for qv in query_vecs:
        top_results = faiss_mgr.search(qv, top_k=total_vecs)
        for r in top_results:
            fname = r["metadata"].get("filename", "")
            scr = r["score"]
            if fname and (fname not in file_scores or scr > file_scores[fname]["score"]):
                file_scores[fname] = {"filename": fname, "score": scr}

    ranked = sorted(file_scores.values(), key=lambda x: x["score"], reverse=True)
    top_sources = [
        {"filename": c["filename"], "semantic": round(c["score"], 4), "rank": i + 1}
        for i, c in enumerate(ranked[:10])
    ]

    return {
        "top_sources": top_sources,
        "total_candidates": len(ranked),
        "filename": file_name,
    }


@app.post("/api/library/source-text")
async def get_library_source_text(data: dict):
    filename = data.get("filename", "")
    if not filename:
        raise HTTPException(400, "Missing filename")
    safe_name = _safe_name(filename)
    basename = Path(safe_name).stem
    text = load_text(basename)
    if text is None:
        fpath = DOC_STORE / safe_name
        if not fpath.exists():
            raise HTTPException(404, "File not found in library")
        text = extract_text(str(fpath))
    return {"filename": safe_name, "content": text}


@app.post("/api/report/export-two")
async def export_pairwise_report(data: dict):
    session = get_session()
    text_a = data.get("file_a_content", "")
    text_b = data.get("file_b_content", "")
    name_a = data.get("file_a_name", "File A")
    name_b = data.get("file_b_name", "File B")
    lang = data.get("language", "auto")

    if lang == "auto":
        lang_a = detect_language(text_a)
        lang_b = detect_language(text_b)
        lang = lang_a if lang_a == lang_b else "en"

    final_score, s1, s2, s3 = compute_ensemble_score(text_a, text_b, lang, session)
    matches = find_matching_positions(text_a, text_b, lang, session)
    coverage_a, coverage_b = compute_coverage(text_a, text_b, matches)
    level = classify_plagiarism(final_score, s1, s2, s3)

    result_data = {
        "final_score": final_score,
        "score_1": s1,
        "score_2": s2,
        "score_3": s3,
        "coverage_a": coverage_a,
        "coverage_b": coverage_b,
        "level": level,
        "matches": matches,
        "file_a": name_a,
        "file_b": name_b,
    }

    fname_a = _safe_name(name_a)
    fname_b = _safe_name(name_b)
    out_name = f"report_{Path(fname_a).stem}_vs_{Path(fname_b).stem}.pdf"
    out_path = REPORTS_DIR / out_name
    generate_pairwise_report(fname_a, fname_b, result_data, str(out_path))

    return {"filename": out_name, "path": str(out_path)}


@app.post("/api/report/export-library")
async def export_library_report(data: dict):
    session = get_session()
    file_content = data.get("file_content", "")
    file_name = data.get("file_name", "document")
    lang = data.get("language", "auto")

    if lang == "auto":
        lang = detect_language(file_content)

    model = session.get_semantic_model(lang)
    # FIX: encode query theo chunk
    query_chunks = chunk_text(file_content, max_words=256, overlap=50)
    query_vecs = model.encode(query_chunks, convert_to_numpy=True)

    faiss_mgr = session.get_faiss_manager(lang)
    file_scores = {}
    for qv in query_vecs:
        top_results = faiss_mgr.search(qv, top_k=20)
        for r in top_results:
            meta = r["metadata"]
            fname = meta.get("filename", "")
            if fname not in file_scores or r["score"] > file_scores[fname]["score"]:
                src_path = meta.get("path", "")
                file_scores[fname] = {
                    "filename": fname,
                    "score": r["score"],
                    "path": src_path,
                }

    detailed = []
    for fname, entry in file_scores.items():
        src_path = entry["path"]
        if not src_path or not os.path.exists(src_path):
            continue
        src_text = extract_text(src_path)
        fs, s1, s2, s3 = compute_ensemble_score(file_content, src_text, lang, session)
        matches = find_matching_positions(file_content, src_text, lang, session)
        cov_a, cov_b = compute_coverage(file_content, src_text, matches)
        detailed.append({
            "filename": fname,
            "score": fs,
            "coverage_a": cov_a,
            "coverage_b": cov_b,
            "level": classify_plagiarism(fs, s1, s2, s3),
        })

    detailed.sort(key=lambda x: x["score"], reverse=True)
    max_score = detailed[0]["score"] if detailed else 0
    level = classify_plagiarism(max_score)

    result_data = {
        "avg_score": max_score,
        "level": level,
        "top_sources": detailed[:5],
    }

    fname = _safe_name(file_name)
    out_name = f"report_{Path(fname).stem}_vs_library.pdf"
    out_path = REPORTS_DIR / out_name
    generate_library_report(fname, result_data, str(out_path))

    return {"filename": out_name, "path": str(out_path)}


@app.get("/api/reports/{filename}")
async def get_report(filename: str):
    # FIX: sanitize filename tránh path traversal
    safe = _safe_name(filename)
    fpath = REPORTS_DIR / safe
    if not fpath.exists():
        raise HTTPException(404, "Report not found")
    return FileResponse(str(fpath), media_type="application/pdf",
                        filename=safe)


@app.post("/api/upload")
async def upload_temp_file(file: UploadFile = File(...)):
    import time
    t0 = time.time()
    safe_orig = _safe_name(file.filename or "unknown")
    file_body = None
    try:
        file_body = await file.read()
    except Exception as e:
        print(f"[UPLOAD] read error: {e}".encode('utf-8', errors='replace').decode('utf-8'))
        return {"filename": safe_orig, "content": ""}
    if not file_body:
        return {"filename": safe_orig, "content": ""}
    # FIX: uuid prefix tránh trùng tên
    unique_name = f"{uuid.uuid4().hex}_{safe_orig}"
    dst = UPLOAD_DIR / unique_name
    try:
        dst.write_bytes(file_body)
        # FIX: extract trong executor để không block event loop
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(executor, extract_text, str(dst))
        dst.unlink(missing_ok=True)
        return {"filename": safe_orig, "content": text}
    except Exception as e:
        err_msg = str(e)
        if dst.exists():
            dst.unlink(missing_ok=True)
        return {"filename": safe_orig, "content": "", "error": err_msg}


@app.get("/api/health")
async def health():
    ready = _models_ready.is_set()
    return {"status": "ok" if ready else "loading", "models_ready": ready}


FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@app.get("/", include_in_schema=False)
async def serve_index():
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>Frontend not found</h1>", status_code=404)
    return HTMLResponse(
        content=index_path.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
