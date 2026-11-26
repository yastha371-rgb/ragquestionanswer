import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.config import settings


class RetrievalService:
    """Service for persistent vector storage and semantic retrieval using ChromaDB."""

    def __init__(self):
        os.makedirs(settings.CHROMA_PATH, exist_ok=True)
        
        try:
            # Modern ChromaDB 0.4+ API
            self.client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
        except Exception:
            # Fallback for older ChromaDB clients
            chroma_settings = ChromaSettings(
                is_persistent=True,
                persist_directory=settings.CHROMA_PATH,
                anonymized_telemetry=False
            )
            self.client = chromadb.Client(chroma_settings)

        self.collection = self.client.get_or_create_collection(
            name=settings.VECTOR_STORE_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        """
        Add or update document chunks and their embeddings in the vector store.
        
        Args:
            documents: List of chunk dictionaries containing 'id', 'text', and 'metadata'
            embeddings: List of embedding vectors corresponding to each chunk
        """
        if not documents:
            return

        ids = [str(doc["id"]) for doc in documents]
        texts = [str(doc["text"]) for doc in documents]
        metadatas = [doc.get("metadata", {}) for doc in documents]

        # Ensure all metadata values are primitive types accepted by Chroma
        sanitized_metadatas = []
        for meta in metadatas:
            clean_meta = {}
            for k, v in meta.items():
                if v is None:
                    continue
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                else:
                    clean_meta[k] = str(v)
            sanitized_metadatas.append(clean_meta)

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=sanitized_metadatas
        )

    def retrieve(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the most semantically relevant document chunks.
        
        Args:
            query_embedding: Embedding of the inquiry
            top_k: Number of chunks to retrieve
            
        Returns:
            List of retrieved chunks with text, metadata, and normalized relevance scores
        """
        # Guard against query on empty collection
        if self.collection.count() == 0:
            return []

        # Bound top_k by collection count
        k = min(top_k, self.collection.count())
        if k <= 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )

        retrieved_docs = []
        if results and results.get("documents") and len(results["documents"]) > 0:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
            distances = results["distances"][0] if results.get("distances") else [0.0] * len(docs)

            for text, meta, dist in zip(docs, metas, distances):
                # Cosine distance to similarity: similarity = 1 - distance
                similarity_score = 1.0 - float(dist)
                normalized_score = max(0.0, min(1.0, round(similarity_score, 4)))

                retrieved_docs.append({
                    "text": text,
                    "metadata": meta,
                    "relevance_score": normalized_score
                })

        # Sort descending by relevance score
        retrieved_docs.sort(key=lambda x: x["relevance_score"], reverse=True)
        return retrieved_docs

    def delete_document(self, doc_id: str) -> None:
        """Delete all chunk embeddings for a specific document ID."""
        try:
            self.collection.delete(where={"doc_id": doc_id})
        except Exception:
            # Fallback search by ID pattern
            results = self.collection.get(where={"doc_id": doc_id})
            if results and results.get("ids"):
                self.collection.delete(ids=results["ids"])

    def list_documents(self) -> List[Dict[str, Any]]:
        """Return list of distinct documents present in the vector store."""
        all_items = self.collection.get()
        doc_ids = set()
        documents = {}

        for meta in all_items.get("metadatas", []):
            if not meta:
                continue
            doc_id = meta.get("doc_id")
            if doc_id and doc_id not in doc_ids:
                doc_ids.add(doc_id)
                documents[doc_id] = meta

        return list(documents.values())

    def get_count(self) -> int:
        """Return total number of vectors in collection."""
        try:
            return self.collection.count()
        except Exception:
            return 0
