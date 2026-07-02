"""
FAISS Tool: semantic search over catalog product descriptions.

Handles fuzzy/topical asks the Metadata Tool can't ("something for
strategic thinking and influencing style", "safety-critical frontline
reliability"). Embeddings are produced locally with sentence-transformers
so no external embedding API/key is required. The index is built once and
cached to disk; rebuild by deleting the cache file or bumping CACHE_VERSION.
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np

from app.core.catalog import Catalog, Product

CACHE_VERSION = 1
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class FaissTool:
    name = "semantic_search"
    description = (
        "Semantic/similarity search over product descriptions and categories. "
        "Use this for fuzzy, topical, or competency-language requests that don't "
        "map cleanly onto exact catalog fields."
    )

    def __init__(self, catalog: Catalog, cache_dir: Path = CACHE_DIR):
        self.catalog = catalog
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._index = None
        self._id_order: list[str] = []
        self._build_or_load()

    # -- lazy heavy imports so importing this module doesn't require
    #    sentence-transformers/faiss unless the tool is actually used --
    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(EMBED_MODEL_NAME)
        return self._model

    def _cache_key(self) -> str:
        ids_blob = "|".join(p.entity_id for p in self.catalog.products)
        digest = hashlib.sha256(ids_blob.encode()).hexdigest()[:16]
        return f"faiss_v{CACHE_VERSION}_{digest}"

    def _build_or_load(self) -> None:
        import faiss

        cache_path = self.cache_dir / f"{self._cache_key()}.pkl"
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                payload = pickle.load(f)
            self._index = faiss.deserialize_index(payload["index_bytes"])
            self._id_order = payload["id_order"]
            return

        model = self._get_model()
        texts = [p.searchable_text() for p in self.catalog.products]
        embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        embeddings = np.asarray(embeddings, dtype="float32")

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # cosine similarity via normalized inner product
        index.add(embeddings)

        self._index = index
        self._id_order = [p.entity_id for p in self.catalog.products]

        payload = {"index_bytes": faiss.serialize_index(index), "id_order": self._id_order}
        with open(cache_path, "wb") as f:
            pickle.dump(payload, f)

    def run(self, query: str, top_k: int = 10, exclude_ids: set[str] | None = None) -> list[tuple[Product, float]]:
        model = self._get_model()
        q_emb = model.encode([query], normalize_embeddings=True)
        q_emb = np.asarray(q_emb, dtype="float32")

        fetch_k = top_k + (len(exclude_ids) if exclude_ids else 0) + 5
        scores, idxs = self._index.search(q_emb, fetch_k)

        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            entity_id = self._id_order[idx]
            if exclude_ids and entity_id in exclude_ids:
                continue
            product = self.catalog.get_by_id(entity_id)
            if product:
                results.append((product, float(score)))
            if len(results) >= top_k:
                break
        return results
