import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ["TORCH_COMPILE_DISABLE"] = "1"
import pickle
import threading
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


class VietnameseEmbedder:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def encode(self, texts, convert_to_numpy=True):
        import torch
        import numpy as np
        embeddings = []
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    text, return_tensors="pt",
                    padding=True, truncation=True, max_length=256
                )
                outputs = self.model(**inputs)
                hidden = outputs.last_hidden_state
                attention = inputs["attention_mask"]
                mask = attention.unsqueeze(-1).float()
                masked = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
                embeddings.append(masked.numpy())
        return np.vstack(embeddings) if convert_to_numpy else embeddings


class AppSession:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    obj = super().__new__(cls)
                    obj._initialized = False
                    cls._instance = obj
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        # --- Stage 1: English models (PAN-PC-11) ---
        self.tfidf_vectorizer_en = None
        self.bert_model_en = None          # fine-tuned multilingual BERT
        self.ensemble_model_en = None

        # --- Stage 2: Vietnamese models (ViSP) ---
        self.tfidf_vectorizer_vi = None
        self.phobert_tokenizer = None
        self.phobert_model_pt = None       # fine-tuned PhoBERT
        self.ensemble_model_vi = None

        # Fallback (original models)
        self.bert_model_fallback = None
        self.models_loaded = False
        self.faiss_managers = {}
        self.document_store = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "database", "document_store"
        )
        os.makedirs(self.document_store, exist_ok=True)

    def load_semantic_models(self, progress_callback=None):
        if self.models_loaded:
            if progress_callback:
                progress_callback(100, "Đã tải xong mô hình.")
            return

        try:
            if progress_callback:
                progress_callback(5, "Đang tải mô hình cho tiếng Anh...")

            # === Load TF-IDF EN ===
            tfidf_path_en = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
            if os.path.exists(tfidf_path_en):
                with open(tfidf_path_en, "rb") as f:
                    self.tfidf_vectorizer_en = pickle.load(f)
                if progress_callback:
                    progress_callback(10, "Đã tải TF-IDF vectorizer (EN).")
            else:
                self.tfidf_vectorizer_en = TfidfVectorizer()
                if progress_callback:
                    progress_callback(10, "Không tìm thấy tfidf_vectorizer.pkl, dùng mặc định.")

            # === Load fine-tuned multilingual BERT (cross-encoder ưu tiên) ===
            if progress_callback:
                progress_callback(15, "Đang tải model cho tiếng Anh...")
            self.cross_encoder_en = None
            cross_path = os.path.join(MODELS_DIR, "cross_encoder_finetuned")
            from sentence_transformers import CrossEncoder
            if os.path.exists(cross_path):
                self.cross_encoder_en = CrossEncoder(cross_path, device="cpu")
                self.bert_model_en = None
                if progress_callback:
                    progress_callback(30, "Đã tải cross-encoder EN (cao nhất).")
            else:
                bert_path = os.path.join(MODELS_DIR, "multilingual_bert_finetuned")
                if os.path.exists(bert_path):
                    from transformers import AutoModel, AutoTokenizer
                    bert_tokenizer = AutoTokenizer.from_pretrained(bert_path)
                    bert_model = AutoModel.from_pretrained(bert_path)
                    self.bert_model_en = VietnameseEmbedder(bert_tokenizer, bert_model)
                    if progress_callback:
                        progress_callback(30, "Đã tải multilingual BERT fine-tuned.")
                else:
                    from sentence_transformers import SentenceTransformer
                    self.bert_model_en = SentenceTransformer(
                        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
                    )
                    if progress_callback:
                        progress_callback(30, "Dùng default Sentence-BERT.")

            # === Load TF-IDF VI ===
            if progress_callback:
                progress_callback(35, "Đang tải mô hình cho tiếng Việt...")
            tfidf_path_vi = os.path.join(MODELS_DIR, "tfidf_vectorizer_vi.pkl")
            if os.path.exists(tfidf_path_vi):
                with open(tfidf_path_vi, "rb") as f:
                    self.tfidf_vectorizer_vi = pickle.load(f)
                if progress_callback:
                    progress_callback(40, "Đã tải TF-IDF vectorizer (VI).")
            else:
                self.tfidf_vectorizer_vi = TfidfVectorizer()

            # === Load fine-tuned PhoBERT ===
            if progress_callback:
                progress_callback(45, "Đang tải PhoBERT fine-tuned...")
            from transformers import AutoTokenizer, AutoModel
            phobert_path = os.path.join(MODELS_DIR, "phobert_finetuned")
            if os.path.exists(phobert_path):
                self.phobert_tokenizer = AutoTokenizer.from_pretrained(phobert_path)
                self.phobert_model_pt = AutoModel.from_pretrained(phobert_path)
                if progress_callback:
                    progress_callback(60, "Đã tải PhoBERT fine-tuned.")
            else:
                # Fallback
                self.phobert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
                self.phobert_model_pt = AutoModel.from_pretrained("vinai/phobert-base")
                if progress_callback:
                    progress_callback(60, "Không tìm thấy fine-tuned, dùng phobert-base.")

            # === Load ensemble models ===
            if progress_callback:
                progress_callback(70, "Đang tải ensemble models...")

            en_path = os.path.join(MODELS_DIR, "ensemble_model_pan.pkl")
            if os.path.exists(en_path):
                with open(en_path, "rb") as f:
                    self.ensemble_model_en = pickle.load(f)
                if progress_callback:
                    progress_callback(75, f"Đã tải ensemble EN (acc: {self.ensemble_model_en.get('accuracy', 0)*100:.1f}%)")

            vi_path = os.path.join(MODELS_DIR, "ensemble_model_vi.pkl")
            if os.path.exists(vi_path):
                with open(vi_path, "rb") as f:
                    self.ensemble_model_vi = pickle.load(f)
                if progress_callback:
                    progress_callback(80, f"Đã tải ensemble VI (acc: {self.ensemble_model_vi.get('accuracy', 0)*100:.1f}%)")

            # === FAISS ===
            if progress_callback:
                progress_callback(85, "Đang tải FAISS indexes...")
            from core.faiss_manager import FAISSManager
            self.faiss_managers["en"] = FAISSManager("en")
            self.faiss_managers["vi"] = FAISSManager("vi")

            self.models_loaded = True
            if progress_callback:
                progress_callback(100, "Sẵn sàng!")

        except Exception as e:
            if progress_callback:
                progress_callback(0, f"Lỗi: {str(e)}")
            raise

    def get_tfidf_vectorizer(self, language):
        if language == "vi" and self.tfidf_vectorizer_vi is not None:
            return self.tfidf_vectorizer_vi
        return self.tfidf_vectorizer_en

    def get_semantic_model(self, language):
        if language == "vi":
            return VietnameseEmbedder(self.phobert_tokenizer, self.phobert_model_pt)
        return self.bert_model_en

    def get_ensemble_model(self, language):
        if language == "vi" and self.ensemble_model_vi is not None:
            return self.ensemble_model_vi
        return self.ensemble_model_en

    def get_faiss_manager(self, language):
        if language not in self.faiss_managers:
            from core.faiss_manager import FAISSManager
            self.faiss_managers[language] = FAISSManager(language)
        return self.faiss_managers[language]
