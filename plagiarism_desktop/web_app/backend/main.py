import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
import sys
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from core.extractor import extract_text, detect_language, chunk_text, count_words
from core.similarity import compute_ensemble_score, classify_plagiarism
from core.highlighter import find_matching_positions
from core.report_generator import generate_pairwise_report, generate_library_report
from core.faiss_manager import FAISSManager
from gui.session import AppSession

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
            if f.suffix.lower() in (".pdf", ".docx", ".txt"):
                size_mb = f.stat().st_size / (1024 * 1024)
                base = f.stem
                npy_path = VEC_STORE / f"{base}.npy"
                files.append({
                    "name": f.name,
                    "size_mb": round(size_mb, 2),
                    "encoded": npy_path.exists(),
                    "language": _detect_file_lang(f),
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
        dst = DOC_STORE / f.filename
        if dst.exists():
            continue
        content = await f.read()
        dst.write_bytes(content)
        saved.append(f.filename)
    return {"saved": saved}


@app.post("/api/library/delete")
async def delete_library_files(data: dict):
    names = data.get("names", [])
    deleted = []
    for name in names:
        fpath = DOC_STORE / name
        if fpath.exists():
            fpath.unlink()
            deleted.append(name)
        base = Path(name).stem
        npy = VEC_STORE / f"{base}.npy"
        if npy.exists():
            npy.unlink()
    _rebuild_all_indexes()
    return {"deleted": deleted}


def _rebuild_all_indexes():
    try:
        session = get_session()
        for lang in ["en", "vi"]:
            faiss_mgr = FAISSManager(lang)
            for fpath in DOC_STORE.iterdir():
                if fpath.suffix.lower() not in (".pdf", ".docx", ".txt"):
                    continue
                base = fpath.stem
                npy_path = VEC_STORE / f"{base}.npy"
                if not npy_path.exists():
                    continue
                import numpy as np
                vec = np.load(str(npy_path))
                faiss_mgr.add_item(vec, {
                    "filename": fpath.name,
                    "path": str(fpath),
                    "language": lang,
                })
            faiss_mgr.save()
            session.faiss_managers[lang] = faiss_mgr
    except Exception as e:
        print(f"Rebuild index error: {e}")


@app.post("/api/library/encode-all")
async def encode_all_library_files():
    session = get_session()
    for fpath in DOC_STORE.iterdir():
        if fpath.suffix.lower() not in (".pdf", ".docx", ".txt"):
            continue
        base = fpath.stem
        npy_path = VEC_STORE / f"{base}.npy"
        if npy_path.exists():
            continue
        try:
            text = extract_text(str(fpath))
            lang = detect_language(text)
            model = session.get_semantic_model(lang)
            vec = model.encode([text], convert_to_numpy=True)[0]
            import numpy as np
            np.save(str(npy_path), vec)
            faiss_mgr = session.get_faiss_manager(lang)
            faiss_mgr.add_item(vec, {
                "filename": fpath.name,
                "path": str(fpath),
                "language": lang,
                "chunks": len(chunk_text(text)),
            })
            faiss_mgr.save()
        except Exception as e:
            print(f"Encode error {fpath.name}: {e}")
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

    final_score, s1, s2, s3 = compute_ensemble_score(text_a, text_b, lang, session)
    matches = find_matching_positions(text_a, text_b, lang, session)
    level = classify_plagiarism(final_score, s1, s2, s3)

    if s3 > 0.9 or s1 > 0.9:
        weight_mode = "Paraphrase: Ngữ nghĩa×0.5 + Từ khóa×0.3 + Nguyên văn×0.2"
    else:
        weight_mode = "Mặc định: Nguyên văn×0.5 + Từ khóa×0.3 + Ngữ nghĩa×0.2"

    matches_clean = []
    for m in matches:
        matches_clean.append({
            "orig_text": m["orig_text"],
            "susp_text": m["susp_text"],
            "score": round(m["score"], 4),
            "type": m["type"],
            "num_sentences": m["num_sentences"],
            "orig_idx_start": m["orig_idx_start"],
            "orig_idx_end": m["orig_idx_end"],
            "susp_idx_start": m["susp_idx_start"],
            "susp_idx_end": m["susp_idx_end"],
            "is_common_knowledge": m.get("is_common_knowledge", False),
        })

    return {
        "final_score": round(final_score, 4),
        "score_1": round(s1, 4),
        "score_2": round(s2, 4),
        "score_3": round(s3, 4),
        "level": level,
        "matches": matches_clean,
        "language": lang,
        "weight_mode": weight_mode,
        "file_a_name": data.file_a_name,
        "file_b_name": data.file_b_name,
    }


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
    query_vec = model.encode([file_content], convert_to_numpy=True)[0]

    faiss_mgr = session.get_faiss_manager(lang)
    top_results = faiss_mgr.search(query_vec, top_k=20)

    detailed = []
    for r in top_results:
        meta = r["metadata"]
        src_path = meta.get("path", "")
        if not os.path.exists(src_path):
            continue
        src_text = extract_text(src_path)
        fs, s1, s2, s3 = compute_ensemble_score(file_content, src_text, lang, session)
        detailed.append({
            "filename": meta.get("filename", os.path.basename(src_path)),
            "score": round(fs, 4),
            "s1": round(s1, 4),
            "s2": round(s2, 4),
            "s3": round(s3, 4),
            "level": classify_plagiarism(fs, s1, s2, s3),
        })

    detailed.sort(key=lambda x: x["score"], reverse=True)
    top5 = detailed[:5]
    avg_score = sum(d["score"] for d in detailed) / len(detailed) if detailed else 0
    level = classify_plagiarism(avg_score)

    return {
        "avg_score": round(avg_score, 4),
        "level": level,
        "top_sources": top5,
        "all_sources": detailed,
        "filename": file_name,
    }


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
    level = classify_plagiarism(final_score, s1, s2, s3)

    result_data = {
        "final_score": final_score,
        "score_1": s1,
        "score_2": s2,
        "score_3": s3,
        "level": level,
        "matches": matches,
        "file_a": name_a,
        "file_b": name_b,
    }

    fname_a = Path(name_a).stem
    fname_b = Path(name_b).stem
    out_name = f"report_{fname_a}_vs_{fname_b}.pdf"
    out_path = REPORTS_DIR / out_name
    generate_pairwise_report(name_a, name_b, result_data, str(out_path))

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
    query_vec = model.encode([file_content], convert_to_numpy=True)[0]
    faiss_mgr = session.get_faiss_manager(lang)
    top_results = faiss_mgr.search(query_vec, top_k=20)

    detailed = []
    for r in top_results:
        meta = r["metadata"]
        src_path = meta.get("path", "")
        if not os.path.exists(src_path):
            continue
        src_text = extract_text(src_path)
        fs, s1, s2, s3 = compute_ensemble_score(file_content, src_text, lang, session)
        detailed.append({
            "filename": meta.get("filename", os.path.basename(src_path)),
            "score": fs,
            "level": classify_plagiarism(fs, s1, s2, s3),
        })

    detailed.sort(key=lambda x: x["score"], reverse=True)
    avg_score = sum(d["score"] for d in detailed) / len(detailed) if detailed else 0
    level = classify_plagiarism(avg_score)

    result_data = {
        "avg_score": avg_score,
        "level": level,
        "top_sources": detailed[:5],
    }

    fname = Path(file_name).stem
    out_name = f"report_{fname}_vs_library.pdf"
    out_path = REPORTS_DIR / out_name
    generate_library_report(file_name, result_data, str(out_path))

    return {"filename": out_name, "path": str(out_path)}


@app.get("/api/reports/{filename}")
async def get_report(filename: str):
    fpath = REPORTS_DIR / filename
    if not fpath.exists():
        raise HTTPException(404, "Report not found")
    return FileResponse(str(fpath), media_type="application/pdf",
                        filename=filename)


@app.post("/api/upload")
async def upload_temp_file(file: UploadFile = File(...)):
    import time
    t0 = time.time()
    print(f"[UPLOAD] start file={file.filename}")
    try:
        content = await file.read()
        print(f"[UPLOAD] read {len(content)} bytes in {time.time()-t0:.2f}s")
        dst = UPLOAD_DIR / file.filename
        dst.write_bytes(content)
        text = extract_text(str(dst))
        dst.unlink(missing_ok=True)
        print(f"[UPLOAD] extracted {len(text)} chars in {time.time()-t0:.2f}s")
        return {"filename": file.filename, "content": text}
    except Exception as e:
        print(f"[UPLOAD] error: {e}")
        raise HTTPException(500, str(e))


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
