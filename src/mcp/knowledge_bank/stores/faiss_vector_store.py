"""
FAISS-based ingestion handler for LangChain Documents.
- Splits text
- Embeds with a local Sentence-Transformer (free/offline)
- Stores vectors in a FAISS index with metadata
- Saves/loads the index from disk

Usage:
    from langchain.schema import Document
    docs = [Document(page_content="Hello world", metadata={"source": "hello.md"})]

    handler = FAISSIngestionHandler(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=800,
        chunk_overlap=150,
        normalize_embeddings=True,
    )

    # Build a new index from documents
    handler.ingest(docs)
    handler.save_local("./vectorstore/faiss_db")

    # Reload later
    handler2 = FAISSIngestionHandler.load_local("./vectorstore/faiss_db")
    print(handler2.similarity_search("world", k=3))
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Iterable, Dict, Any, Tuple
import os
import uuid

from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


DEFAULT_SEPARATORS = [
    "\n\n", "\n", "。", "！", "？", 
    ". ", "! ", "? ", ", ", "; ", ": ",
    "- ", "— ", " • ", "\t", " "
]


@dataclass
class FAISSIngestionConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    normalize_embeddings: bool = True
    chunk_size: int = 800
    chunk_overlap: int = 150
    separators: Optional[List[str]] = None
    device: Optional[str] = None  # e.g., "cpu", "cuda", "mps"
    batch_size: int = 64


class FAISSIngestionHandler:
    """End-to-end FAISS ingestion for LangChain Documents."""

    def __init__(
            self, 
            model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
            normalize_embeddings: bool = True,
            chunk_size: int = 800,
            chunk_overlap: int = 150,
            separators: Optional[List[str]] = None,
            device: Optional[str] = None,
            batch_size: int = 64
        ) -> None:
        self.config = FAISSIngestionConfig(
            model_name=model_name,
            normalize_embeddings=normalize_embeddings,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators or DEFAULT_SEPARATORS,
            device=device,
            batch_size=batch_size,
        )
        self._init_components()
        self._vs: Optional[FAISS] = None

    # ------------- public API -------------
    @property
    def vectorstore(self) -> FAISS:
        if self._vs is None:
            raise RuntimeError("Vectorstore is not initialized. Call ingest(...) or load_local(...).")
        return self._vs

    def ingest(self, documents: List[Document]) -> int:
        """Split, embed, and create a new FAISS index from the given documents.
        Returns the number of chunks added.
        """
        chunks = self._split_documents(documents)
        ids = self._assign_chunk_ids(chunks)
        self._vs = FAISS.from_documents(
            chunks,
            self._embeddings,
            ids=ids,
        )
        return len(chunks)

    def add(self, documents: List[Document]) -> int:
        """Split, embed, and add to the existing FAISS index. Creates one if missing."""
        chunks = self._split_documents(documents)
        ids = self._assign_chunk_ids(chunks)
        if self._vs is None:
            self._vs = FAISS.from_documents(chunks, self._embeddings, ids=ids)
        else:
            self._vs.add_documents(chunks, ids=ids)
        return len(chunks)

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        return self.vectorstore.similarity_search(query, k=k)

    def similarity_search_with_score(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        return self.vectorstore.similarity_search_with_score(query, k=k)

    def save_local(self, path: str) -> None:
        if self._vs is None:
            raise RuntimeError("Nothing to save: vectorstore is empty.")
        os.makedirs(path, exist_ok=True)
        self._vs.save_local(path)

    @classmethod
    def load_local(cls, path: str) -> "FAISSIngestionHandler":
        """Load an existing FAISS index and embeddings from disk."""
        handler = cls()
        handler._vs = FAISS.load_local(
            path,
            handler._embeddings,
            allow_dangerous_deserialization=True,
        )
        return handler

    def rebuild_excluding_sources(self, excluded_sources: Iterable[str]) -> int:
        """Rebuilds the index without any chunks whose metadata['source'] is in excluded_sources.
        Returns number of chunks in the rebuilt index.
        """
        if self._vs is None:
            return 0
        docs: List[Document] = list(self._vs.docstore._dict.values())  # type: ignore[attr-defined]
        filtered = [d for d in docs if d.metadata.get("source") not in set(excluded_sources)]
        chunks = self._split_documents(filtered)
        ids = self._assign_chunk_ids(chunks)
        self._vs = FAISS.from_documents(chunks, self._embeddings, ids=ids)
        return len(chunks)

    # ------------- internal helpers -------------
    def _init_components(self) -> None:
        # Embeddings: local, free model with optional normalization & device
        encode_kwargs = {"normalize_embeddings": self.config.normalize_embeddings}
        if self.config.device:
            encode_kwargs["device"] = self.config.device
        # Some versions expect model_kwargs/encode_kwargs split; we keep it simple here.
        self._embeddings = HuggingFaceEmbeddings(
            model_name=self.config.model_name,
            encode_kwargs=encode_kwargs,
        )
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=self.config.separators,
            add_start_index=True,
        )

    def _split_documents(self, documents: List[Document]) -> List[Document]:
        # Ensure we carry useful metadata into chunks
        base_fields = ["source", "path", "file_name", "doc_id", "title"]
        chunks = self._splitter.split_documents(documents)
        for i, d in enumerate(chunks):
            # carry through base fields if present
            meta = {k: v for k, v in d.metadata.items() if k in base_fields}
            # assign stable doc_id if missing
            if "doc_id" not in meta:
                meta["doc_id"] = d.metadata.get("doc_id") or str(uuid.uuid4())
            meta.update({
                "chunk_id": i,
                "chunk_size": len(d.page_content),
            })
            d.metadata = {**d.metadata, **meta}
        return chunks

    def _assign_chunk_ids(self, chunks: List[Document]) -> List[str]:
        ids: List[str] = []
        for d in chunks:
            # Stable-ish ID combining doc and chunk ids
            doc_id = d.metadata.get("doc_id") or "unknown"
            chunk_id = d.metadata.get("chunk_id")
            ids.append(f"{doc_id}::chunk::{chunk_id}")
        return ids


# ---------- Simple smoke test ----------
if __name__ == "__main__":
    # Minimal example to check everything wires up
    texts = [
        "LangChain makes it easy to build LLM applications.",
        "FAISS is a fast vector database for similarity search.",
        "Sentence-Transformers provide high-quality embeddings.",
    ]
    docs = [Document(page_content=t, metadata={"source": f"mem_{i}.md"}) for i, t in enumerate(texts)]

    handler = FAISSIngestionHandler(chunk_size=200, chunk_overlap=20)
    added = handler.ingest(docs)
    print(f"Chunks added: {added}")

    res = handler.similarity_search("What helps with similarity search?", k=2)
    for r in res:
        print("- ", r.page_content[:80], "...", r.metadata)

    handler.save_local("./stores/_faiss_demo")
    handler2 = FAISSIngestionHandler.load_local("./stores/_faiss_demo")
    res2 = handler2.similarity_search_with_score("How do I embed text?", k=2)
    for r, score in res2:
        print(f"score={score:.4f} :: {r.metadata.get('source')} :: {r.page_content[:25]}...")
