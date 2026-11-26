import re
import hashlib
from typing import List, Dict, Any, Optional
from langchain_google_genai import GoogleGenerativeAIEmbeddings

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.config import settings


class EmbeddingService:
    """Service for splitting text and generating vector embeddings via Google Gemini."""

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.embeddings = None
        self._ensure_client()

    def _ensure_client(self) -> None:
        """Ensure embedding client is instantiated with the active key and model."""
        key = settings.api_key if settings.api_key else "dummy-key-for-init"
        model_name = settings.EMBEDDING_MODEL or "models/gemini-embedding-001"

        if (
            self.embeddings is None
            or getattr(self.embeddings, "model", None) != model_name
            or getattr(self.embeddings, "google_api_key", None) != key
        ):
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=key
            )

    def split_documents(self, text: str, doc_id: str, title: str = "") -> List[Dict[str, Any]]:
        """
        Split document into chunks with metadata and page attribution.
        """
        chunks = self.text_splitter.split_text(text)
        chunk_documents = []
        
        page_pattern = re.compile(r"--- Page (\d+) ---")
        current_page: Optional[int] = 1

        for idx, chunk in enumerate(chunks):
            page_matches = page_pattern.findall(chunk)
            if page_matches:
                current_page = int(page_matches[-1])
            
            chunk_id = f"{doc_id}_chunk_{idx}"
            chunk_hash = hashlib.md5(chunk.encode("utf-8")).hexdigest()

            chunk_documents.append({
                "id": chunk_id,
                "text": chunk,
                "metadata": {
                    "doc_id": doc_id,
                    "title": title or doc_id,
                    "chunk_index": idx,
                    "chunk_hash": chunk_hash,
                    "total_chunks": len(chunks),
                    "page_number": current_page
                }
            })

        return chunk_documents

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding vector for a single query text."""
        self._ensure_client()
        return self.embeddings.embed_query(text)

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embedding vectors for multiple text chunks in a batch."""
        self._ensure_client()
        if not texts:
            return []
        return self.embeddings.embed_documents(texts)
