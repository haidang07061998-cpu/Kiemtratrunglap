import os
import numpy as np
import faiss
import pickle

VECTOR_STORE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "database", "vector_store")


class FAISSManager:
    def __init__(self, language):
        self.language = language
        self.index_path = os.path.join(VECTOR_STORE, f"{language}_index.faiss")
        self.meta_path = os.path.join(VECTOR_STORE, f"{language}_metadata.pkl")
        self.dimension = None
        self.index = None
        self.metadata = []
        self._load_or_create()

    def _load_or_create(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            self.dimension = self.index.d
            if os.path.exists(self.meta_path):
                with open(self.meta_path, "rb") as f:
                    self.metadata = pickle.load(f)
        else:
            self.index = None
            self.metadata = []

    def set_dimension(self, dim):
        self.dimension = dim
        if self.index is None:
            self.index = faiss.IndexFlatIP(dim)

    def add_item(self, vector, meta):
        if self.index is None:
            self.set_dimension(vector.shape[0])
        vec = np.array([vector]).astype(np.float32)
        faiss.normalize_L2(vec)
        self.index.add(vec)
        self.metadata.append(meta)

    def add_items(self, vectors, metas):
        for v, m in zip(vectors, metas):
            self.add_item(v, m)

    def search(self, query_vector, top_k=20):
        if self.index is None or self.index.ntotal == 0:
            return []
        q = np.array([query_vector]).astype(np.float32)
        faiss.normalize_L2(q)
        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(q, k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.metadata):
                results.append({
                    "score": float(dist),
                    "metadata": self.metadata[idx],
                })
        return results

    def save(self):
        os.makedirs(VECTOR_STORE, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

    # FIX: reset index — xóa toàn bộ vector và metadata
    def reset(self):
        self.index = None
        self.metadata = []
        self.dimension = None
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.meta_path):
            os.remove(self.meta_path)

    def get_total_count(self):
        return self.index.ntotal if self.index else 0
