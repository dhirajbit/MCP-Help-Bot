"""Chunk articles and store in a lightweight BM25 search index.

Pure-Python BM25 index that fits comfortably in small deployment tiers.
"""

import json
import logging
import math
import os
import re
import threading
from collections import Counter

import config
from sync.notion_sync import Article

logger = logging.getLogger(__name__)

# -- Chunking --

_APPROX_CHARS_PER_TOKEN = 4
CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50
CHUNK_SIZE_CHARS = CHUNK_SIZE_TOKENS * _APPROX_CHARS_PER_TOKEN
CHUNK_OVERLAP_CHARS = CHUNK_OVERLAP_TOKENS * _APPROX_CHARS_PER_TOKEN


def _split_into_sentences(text: str) -> list[str]:
    paragraphs = re.split(r"\n{2,}", text)
    sentences: list[str] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        parts = re.split(r"(?<=[.!?])\s+", para)
        sentences.extend(p.strip() for p in parts if p.strip())
    return sentences


def chunk_article(article: Article) -> list[dict]:
    sentences = _split_into_sentences(article.content)
    if not sentences:
        return []

    chunks: list[dict] = []
    current_chunk: list[str] = []
    current_len = 0
    chunk_idx = 0

    for sentence in sentences:
        sent_len = len(sentence)

        if current_len + sent_len > CHUNK_SIZE_CHARS and current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(
                {
                    "id": f"{article.id}::chunk_{chunk_idx}",
                    "text": chunk_text,
                    "metadata": {
                        "article_id": article.id,
                        "article_title": article.title,
                        "collection": article.collection,
                        "url": article.url,
                        "chunk_index": chunk_idx,
                    },
                }
            )
            chunk_idx += 1

            overlap_chars = 0
            overlap_start = len(current_chunk)
            for i in range(len(current_chunk) - 1, -1, -1):
                overlap_chars += len(current_chunk[i])
                if overlap_chars >= CHUNK_OVERLAP_CHARS:
                    overlap_start = i
                    break
            current_chunk = current_chunk[overlap_start:]
            current_len = sum(len(s) for s in current_chunk)

        current_chunk.append(sentence)
        current_len += sent_len

    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append(
            {
                "id": f"{article.id}::chunk_{chunk_idx}",
                "text": chunk_text,
                "metadata": {
                    "article_id": article.id,
                    "article_title": article.title,
                    "collection": article.collection,
                    "url": article.url,
                    "chunk_index": chunk_idx,
                },
            }
        )

    return chunks


# -- BM25 Search Index --

_STOP_WORDS = frozenset(
    "a an the is are was were be been being have has had do does did will "
    "would shall should may might can could of in to for on with at by from "
    "as into through during before after above below between out off over "
    "under again further then once here there when where why how all each "
    "every both few more most other some such no nor not only own same so "
    "than too very and but if or because until while about this that these "
    "those it its i me my we our you your he him his she her they them their "
    "what which who whom".split()
)

_K1 = 1.5
_B = 0.75


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics, remove stop words."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


class _BM25Data:
    """Immutable snapshot of BM25 index data — enables atomic swap."""
    __slots__ = (
        "ids", "documents", "metadatas",
        "doc_term_freqs", "doc_lengths", "avg_dl", "doc_freqs", "n_docs",
    )

    def __init__(
        self,
        ids: list[str] | None = None,
        documents: list[str] | None = None,
        metadatas: list[dict] | None = None,
        doc_term_freqs: list[dict[str, int]] | None = None,
        doc_lengths: list[int] | None = None,
        avg_dl: float = 0.0,
        doc_freqs: dict[str, int] | None = None,
        n_docs: int = 0,
    ):
        self.ids = ids or []
        self.documents = documents or []
        self.metadatas = metadatas or []
        self.doc_term_freqs = doc_term_freqs or []
        self.doc_lengths = doc_lengths or []
        self.avg_dl = avg_dl
        self.doc_freqs = doc_freqs or {}
        self.n_docs = n_docs


class BM25Index:
    """Pure-Python BM25 search index persisted as a JSON file."""

    def __init__(self, persist_path: str):
        self._persist_path = persist_path
        self._data = _BM25Data()
        self._loaded = False
        self._load_lock = threading.Lock()

    def _ensure_loaded(self):
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            if os.path.exists(self._persist_path):
                try:
                    with open(self._persist_path, "r") as f:
                        raw = json.load(f)
                    self._data = _BM25Data(
                        ids=raw["ids"],
                        documents=raw["documents"],
                        metadatas=raw["metadatas"],
                        doc_term_freqs=[
                            {k: v for k, v in d.items()} for d in raw["doc_term_freqs"]
                        ],
                        doc_lengths=raw["doc_lengths"],
                        avg_dl=raw["avg_dl"],
                        doc_freqs=raw["doc_freqs"],
                        n_docs=raw["n_docs"],
                    )
                    logger.info("Loaded BM25 index from disk (%d docs)", self._data.n_docs)
                except Exception:
                    logger.warning("Failed to load BM25 index — starting empty", exc_info=True)
            self._loaded = True

    def _save(self, data: _BM25Data):
        os.makedirs(os.path.dirname(self._persist_path), exist_ok=True)
        with open(self._persist_path, "w") as f:
            json.dump(
                {
                    "ids": data.ids,
                    "documents": data.documents,
                    "metadatas": data.metadatas,
                    "doc_term_freqs": data.doc_term_freqs,
                    "doc_lengths": data.doc_lengths,
                    "avg_dl": data.avg_dl,
                    "doc_freqs": data.doc_freqs,
                    "n_docs": data.n_docs,
                },
                f,
            )
        logger.info("Saved BM25 index to disk (%d docs)", data.n_docs)

    def rebuild(self, ids: list[str], documents: list[str], metadatas: list[dict]):
        """Build a new BM25 index and atomically swap it in."""
        n_docs = len(documents)

        doc_term_freqs: list[dict[str, int]] = []
        doc_lengths: list[int] = []
        doc_freqs: dict[str, int] = {}

        for doc in documents:
            tokens = _tokenize(doc)
            tf = dict(Counter(tokens))
            doc_term_freqs.append(tf)
            doc_lengths.append(len(tokens))
            for term in set(tokens):
                doc_freqs[term] = doc_freqs.get(term, 0) + 1

        total_len = sum(doc_lengths)
        avg_dl = total_len / n_docs if n_docs else 0.0

        new_data = _BM25Data(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            doc_term_freqs=doc_term_freqs,
            doc_lengths=doc_lengths,
            avg_dl=avg_dl,
            doc_freqs=doc_freqs,
            n_docs=n_docs,
        )

        self._save(new_data)
        self._data = new_data
        self._loaded = True

    def count(self) -> int:
        self._ensure_loaded()
        return self._data.n_docs

    def query(
        self,
        query_texts: list[str],
        n_results: int = 8,
        **kwargs,
    ) -> dict:
        """BM25-rank documents against the query."""
        self._ensure_loaded()

        d = self._data

        if d.n_docs == 0:
            return {"documents": [[]], "metadatas": [[]], "ids": [[]]}

        query_tokens = _tokenize(query_texts[0])
        if not query_tokens:
            return {"documents": [[]], "metadatas": [[]], "ids": [[]]}

        scores: list[tuple[float, int]] = []

        for idx in range(d.n_docs):
            score = 0.0
            tf_doc = d.doc_term_freqs[idx]
            dl = d.doc_lengths[idx]

            for term in query_tokens:
                if term not in d.doc_freqs:
                    continue
                df = d.doc_freqs[term]
                tf = tf_doc.get(term, 0)
                if tf == 0:
                    continue

                idf = math.log((d.n_docs - df + 0.5) / (df + 0.5) + 1.0)
                numerator = tf * (_K1 + 1)
                denominator = tf + _K1 * (1 - _B + _B * dl / d.avg_dl)
                score += idf * numerator / denominator

            scores.append((score, idx))

        scores.sort(key=lambda x: x[0], reverse=True)
        top = scores[:n_results]

        top_docs = [d.documents[idx] for _, idx in top if _ > 0]
        top_metas = [d.metadatas[idx] for _, idx in top if _ > 0]
        top_ids = [d.ids[idx] for _, idx in top if _ > 0]

        return {
            "documents": [top_docs],
            "metadatas": [top_metas],
            "ids": [top_ids],
        }


# -- Singleton index --

_INDEX_PATH = os.path.join(config.SEARCH_INDEX_DIR, "bm25_index.json")
_index = BM25Index(_INDEX_PATH)


def rebuild_vector_store(articles: list[Article]) -> int:
    """Chunk all articles and rebuild the BM25 search index."""
    logger.info("Rebuilding search index with %d articles", len(articles))

    all_chunks: list[dict] = []
    for article in articles:
        all_chunks.extend(chunk_article(article))

    if not all_chunks:
        logger.warning("No chunks to index — search store is empty")
        return 0

    _index.rebuild(
        ids=[c["id"] for c in all_chunks],
        documents=[c["text"] for c in all_chunks],
        metadatas=[c["metadata"] for c in all_chunks],
    )

    logger.info("Indexed %d chunks", len(all_chunks))
    return len(all_chunks)


def get_collection() -> BM25Index:
    """Return the BM25 index."""
    return _index
