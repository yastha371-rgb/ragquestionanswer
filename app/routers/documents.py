import os
import json
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from app.config import settings
from app.models import DocumentInfo
from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService
from app.services.retrieval_service import RetrievalService
from app.utils.validators import validate_file_extension, validate_file_size

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/documents", tags=["Documents"])

# Re-instantiate services to ensure active configuration
embedding_service = EmbeddingService()
retrieval_service = RetrievalService()

METADATA_FILE = os.path.join(settings.DOCUMENT_STORAGE_PATH, "..", "documents_metadata.json")


def _load_metadata() -> dict:
    """Load metadata from persistent JSON store."""
    try:
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load metadata file: {e}")
    return {}


def _save_metadata(metadata: dict) -> None:
    """Save metadata to persistent JSON store."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(METADATA_FILE)), exist_ok=True)
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata, f, default=str, indent=2)
    except Exception as e:
        logger.warning(f"Could not save metadata file: {e}")


# Initialize in-memory cache synchronized with disk
document_metadata = _load_metadata()


@router.post("/upload", response_model=DocumentInfo)
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None)
) -> DocumentInfo:
    """
    Upload and process an academic document (PDF, TXT, DOCX), extracting and indexing vector embeddings.
    """
    os.makedirs(settings.DOCUMENT_STORAGE_PATH, exist_ok=True)

    # 1. Validate file extension
    if not validate_file_extension(file.filename, settings.ALLOWED_FILE_TYPES):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed types are: {', '.join(settings.ALLOWED_FILE_TYPES)}"
        )

    doc_id = str(uuid.uuid4())
    doc_title = title.strip() if title and title.strip() else file.filename
    clean_filename = f"{doc_id}_{Path(file.filename).name}"
    file_path = os.path.join(settings.DOCUMENT_STORAGE_PATH, clean_filename)

    try:
        # 2. Stream and write file to storage
        content = await file.read()
        
        # Validate size
        is_valid, size_error = validate_file_size(len(content), settings.MAX_FILE_SIZE)
        if not is_valid:
            raise HTTPException(status_code=400, detail=size_error)

        with open(file_path, "wb") as f:
            f.write(content)

        file_size = os.path.getsize(file_path)

        # 3. Extract text content
        file_ext = file.filename.rsplit(".", 1)[-1].lower()
        extracted_text = DocumentProcessor.process_document(file_path, file_ext)

        if not extracted_text or not extracted_text.strip():
            raise HTTPException(
                status_code=400,
                detail="Extracted document content is empty. Please verify the document text content."
            )

        # 4. Split into chunks with document metadata
        chunks = embedding_service.split_documents(
            text=extracted_text,
            doc_id=doc_id,
            title=doc_title
        )

        # 5. Generate embeddings and persist to vector store
        chunk_texts = [chunk["text"] for chunk in chunks]
        embeddings = embedding_service.get_embeddings_batch(chunk_texts)
        retrieval_service.add_documents(chunks, embeddings)

        # 6. Record metadata
        doc_info = {
            "doc_id": doc_id,
            "title": doc_title,
            "description": description,
            "file_name": file.filename,
            "uploaded_at": datetime.now().isoformat(),
            "chunk_count": len(chunks),
            "size_bytes": file_size
        }

        document_metadata[doc_id] = doc_info
        _save_metadata(document_metadata)

        return DocumentInfo(**doc_info)

    except HTTPException:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass
        raise
    except Exception as e:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass
        logger.error(f"Error processing document upload: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {str(e)}"
        )


@router.get("/list", response_model=List[DocumentInfo])
async def list_documents() -> List[DocumentInfo]:
    """Retrieve all indexed academic documents."""
    return [DocumentInfo(**meta) for meta in document_metadata.values()]


@router.get("/{doc_id}", response_model=DocumentInfo)
async def get_document(doc_id: str) -> DocumentInfo:
    """Retrieve metadata for a specific document by its ID."""
    if doc_id not in document_metadata:
        raise HTTPException(status_code=404, detail="Document not found.")
    return DocumentInfo(**document_metadata[doc_id])


@router.delete("/{doc_id}")
async def delete_document(doc_id: str) -> dict:
    """Delete a document, removing its chunks from the vector database and disk."""
    if doc_id not in document_metadata:
        raise HTTPException(status_code=404, detail="Document not found.")

    try:
        # Delete from ChromaDB
        retrieval_service.delete_document(doc_id)

        # Remove local file
        for file_path in Path(settings.DOCUMENT_STORAGE_PATH).glob(f"{doc_id}_*"):
            try:
                os.remove(file_path)
            except OSError as e:
                logger.warning(f"Could not delete physical file {file_path}: {e}")

        # Remove from metadata
        del document_metadata[doc_id]
        _save_metadata(document_metadata)

        return {"message": "Document and associated embeddings successfully deleted."}

    except Exception as e:
        logger.error(f"Error deleting document {doc_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document: {str(e)}"
        )
