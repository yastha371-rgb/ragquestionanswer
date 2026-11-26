# 🎓 Academic Q&A RAG System: Complete Code & Technical Documentation

> **Multi-Format Enterprise RAG Platform supporting `.PDF`, `.DOCX`, `.TXT`, `.CSV`, `.XLSX`, and `.XLS`**  
> **Backend:** FastAPI, LangChain, Google Gemini (`gemini-3.8-flash` & `gemini-embedding-001`), ChromaDB, Redis, PyPDF, python-docx, pandas, openpyxl, xlrd.

---

## 📑 Table of Contents

1. [Architecture & Document Ingestion Pipeline](#1-architecture--document-ingestion-pipeline)
2. [Supported Document Formats](#2-supported-document-formats)
3. [Configuration & Environment (`app/config.py`, `.env`)](#3-configuration--environment)
4. [Data Models (`app/models.py`)](#4-data-models)
5. [Document Processing Engine (`app/services/document_processor.py`)](#5-document-processing-engine)
6. [Vector Embedding Service (`app/services/embedding_service.py`)](#6-vector-embedding-service)
7. [ChromaDB Vector Retrieval (`app/services/retrieval_service.py`)](#7-chromadb-vector-retrieval)
8. [LLM Answer Generation & Citations (`app/services/generation_service.py`)](#8-llm-answer-generation--citations)
9. [Redis Cache Tier (`app/services/cache_service.py`)](#9-redis-cache-tier)
10. [REST API Routers (`app/routers/`)](#10-rest-api-routers)
11. [FastAPI Application & Web UI (`app/main.py`)](#11-fastapi-application--web-ui)
12. [Dependencies (`requirements.txt`)](#12-dependencies)

---

## 1. Architecture & Document Ingestion Pipeline

```
+-----------------------------------------------------------------------------------+
|                           Academic Q&A RAG System                                 |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  [Uploaded Files]                                                                 |
|   ├── .PDF  ──> PyPDF (Extracts pages with page markers for citations)            |
|   ├── .DOCX ──> python-docx (Extracts paragraphs & structured tables)            |
|   ├── .TXT  ──> Native UTF-8 / Latin-1 Stream Reader                             |
|   ├── .CSV  ──> pandas / csv (Formats into structured row-entity records)        |
|   └── .XLSX / .XLS ──> pandas + openpyxl + xlrd (Multi-sheet table serialization)|
|                                                                                   |
|                                       │                                           |
|                                       ▼                                           |
|                     [DocumentProcessor.process_document]                         |
|                                       │                                           |
|                                       ▼                                           |
|                   [RecursiveCharacterTextSplitter Chunking]                      |
|                                       │                                           |
|                                       ▼                                           |
|                [Google Gemini Embeddings: gemini-embedding-001]                  |
|                                       │                                           |
|                                       ▼                                           |
|                   [ChromaDB Persistent Vector Store: Cosine]                     |
|                                                                                   |
|  [User Question]                                                                  |
|   ──> Fast Cache Check (Redis / In-Memory bypass)                                 |
|   ──> Vector Similarity Query (Top-K context retrieval)                           |
|   ──> Google Gemini 3.8 Flash (LCEL Prompt Chain)                                 |
|   ──> Answer with Confidence Score & Document / Page Citations                    |
+-----------------------------------------------------------------------------------+
```

---

## 2. Supported Document Formats

| Format | Library Engine | Extraction Strategy & RAG Formatting |
|---|---|---|
| **`.PDF`** | `pypdf` | Extracts page-by-page text with `--- Page X ---` tags, enabling exact page citations in answers. |
| **`.DOCX`** | `python-docx` | Extracts headings, paragraphs, and iterates through every embedded table row and cell. |
| **`.TXT`** | Native Python | Reads text with UTF-8 encoding and fallback to Latin-1 for legacy research documents. |
| **`.CSV`** | `pandas` + `csv` | Converts tabular datasets into semantic records: `Row N: ColA: Val1 \| ColB: Val2`. |
| **`.XLSX`** | `pandas` + `openpyxl` | Loops through all workbook sheets, retaining sheet names, dimensions, and column relationships. |
| **`.XLS`** | `pandas` + `xlrd` | Supports legacy Excel files, translating each sheet and row into readable entity descriptions. |

---

## 3. Configuration & Environment

### `app/config.py`
```python
import os
from typing import List, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration for Academic Q&A RAG System."""
    
    API_TITLE: str = "Academic Q&A RAG System"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "Enterprise RAG system for academic Q&A powered by Google Gemini"
    DEBUG: bool = False
    
    # Free AI Provider: Google Gemini
    GEMINI_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""
    MODEL_NAME: str = "gemini-3.8-flash"
    EMBEDDING_MODEL: str = "models/gemini-embedding-001"
    
    # Database Configuration
    CHROMA_PATH: str = "./data/chroma_db"
    VECTOR_STORE_NAME: str = "academic_qa"
    
    # Redis Configuration
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    CACHE_TTL: int = 3600  # 1 hour
    
    # Document Configuration (Supporting PDF, DOCX, TXT, CSV, XLSX, XLS)
    DOCUMENT_STORAGE_PATH: str = "./data/documents"
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_FILE_TYPES: Union[List[str], str] = ["pdf", "txt", "docx", "csv", "xlsx", "xls"]
    
    # RAG Configuration
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    RETRIEVAL_TOP_K: int = 5
    MIN_CONFIDENCE_SCORE: float = 0.5

    @property
    def api_key(self) -> str:
        """Returns the configured Gemini API key."""
        return self.GEMINI_API_KEY or self.GOOGLE_API_KEY or os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
    
    @field_validator("ALLOWED_FILE_TYPES", mode="before")
    @classmethod
    def parse_allowed_file_types(cls, v):
        if isinstance(v, str):
            return [ext.strip().lower() for ext in v.split(",") if ext.strip()]
        return [ext.strip().lower() for ext in v]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


settings = Settings()
```

---

## 4. Document Processing Engine

### `app/services/document_processor.py`
```python
import os
import csv
from pathlib import Path
from typing import Tuple, List, Dict, Any
from pypdf import PdfReader
from app.config import settings
from app.utils.validators import validate_file_size


class DocumentProcessor:
    """Service for processing uploaded academic documents across formats (PDF, DOCX, TXT, CSV, XLSX, XLS)."""

    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        """Extract all text from a PDF file with page indicators."""
        text = ""
        try:
            reader = PdfReader(file_path)
            for page_num, page in enumerate(reader.pages):
                extracted = page.extract_text()
                if extracted:
                    text += f"\n--- Page {page_num + 1} ---\n{extracted}"
        except Exception as e:
            raise ValueError(f"Failed to extract text from PDF: {str(e)}")
        return text

    @staticmethod
    def extract_text_from_txt(file_path: str) -> str:
        """Extract text from plain text file with UTF-8 encoding (fallback to latin-1)."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, "r", encoding="latin-1") as f:
                    return f.read()
            except Exception as e:
                raise ValueError(f"Failed to read text file with fallback encoding: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to read text file: {str(e)}")

    @staticmethod
    def extract_text_from_docx(file_path: str) -> str:
        """Extract text paragraphs and tables from DOCX file."""
        try:
            from docx import Document
            doc = Document(file_path)
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            
            # Extract text from tables
            table_texts = []
            for table_idx, table in enumerate(doc.tables):
                table_texts.append(f"\n[Table {table_idx + 1}]")
                for row in table.rows:
                    row_text = " | ".join([cell.text.strip() for cell in row.cells if cell.text.strip()])
                    if row_text:
                        table_texts.append(row_text)
            
            return "\n".join(paragraphs + table_texts)
        except Exception as e:
            raise ValueError(f"Failed to extract text from DOCX: {str(e)}")

    @staticmethod
    def extract_text_from_csv(file_path: str) -> str:
        """Extract and format tabular CSV data into structured, readable document text."""
        try:
            import pandas as pd
            df = pd.read_csv(file_path, encoding_errors="replace")
            
            lines = [
                f"[CSV Dataset: {Path(file_path).name}]",
                f"Total Records: {len(df)} | Columns: {len(df.columns)}",
                f"Column Names: {', '.join([str(c) for c in df.columns])}\n"
            ]
            for idx, row in df.iterrows():
                row_items = [f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])]
                if row_items:
                    lines.append(f"Row {idx + 1}: " + " | ".join(row_items))
            return "\n".join(lines)
        except Exception as e:
            raise ValueError(f"Failed to parse CSV file: {str(e)}")

    @staticmethod
    def extract_text_from_excel(file_path: str) -> str:
        """Extract multi-sheet Excel (.xlsx, .xls) data into structured, readable document text."""
        try:
            import pandas as pd
            with pd.ExcelFile(file_path) as excel_file:
                sheet_names = excel_file.sheet_names
                output_parts = [
                    f"[Excel Workbook: {Path(file_path).name}]",
                    f"Sheets: {', '.join(sheet_names)}\n"
                ]
                for sheet_idx, sheet in enumerate(sheet_names):
                    df = pd.read_excel(excel_file, sheet_name=sheet)
                    output_parts.append(f"--- Sheet: {sheet} (Page {sheet_idx + 1}) ---")
                    output_parts.append(f"Dimensions: {len(df)} rows x {len(df.columns)} columns")
                    output_parts.append(f"Headers: {', '.join([str(c) for c in df.columns])}\n")
                    for row_idx, row in df.iterrows():
                        row_items = [f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])]
                        if row_items:
                            output_parts.append(f"Row {row_idx + 1}: " + " | ".join(row_items))
                    output_parts.append("")
            return "\n".join(output_parts)
        except Exception as e:
            raise ValueError(f"Failed to extract text from Excel file: {str(e)}")

    @staticmethod
    def process_document(file_path: str, file_type: str) -> str:
        """Process document and extract text based on file format."""
        file_type = file_type.lower().lstrip(".")
        if file_type == "pdf":
            return DocumentProcessor.extract_text_from_pdf(file_path)
        elif file_type == "txt":
            return DocumentProcessor.extract_text_from_txt(file_path)
        elif file_type == "docx":
            return DocumentProcessor.extract_text_from_docx(file_path)
        elif file_type == "csv":
            return DocumentProcessor.extract_text_from_csv(file_path)
        elif file_type in ["xlsx", "xls"]:
            return DocumentProcessor.extract_text_from_excel(file_path)
        else:
            raise ValueError(f"Unsupported file type: .{file_type}")
```

---

## 5. Vector Embedding Service

### `app/services/embedding_service.py`
```python
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
        self._ensure_client()
        return self.embeddings.embed_query(text)

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        self._ensure_client()
        if not texts:
            return []
        return self.embeddings.embed_documents(texts)
```

---

## 6. How to Run & Ingest All Supported Documents

1. **Start the FastAPI Server**:
   ```powershell
   cd C:\Users\astha\.gemini\antigravity\scratch\academic-qa-rag
   .\venv\Scripts\Activate.ps1
   python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
   ```

2. **Open the Web UI**:
   Navigate to **`http://127.0.0.1:8000/`**.

3. **Upload any document type**:
   - Select your `.pdf`, `.docx`, `.txt`, `.csv`, `.xlsx`, or `.xls` file.
   - Enter a title.
   - Click **Upload & Vectorize**.

4. **Ask Research & Data Questions**:
   Type questions like:
   - *"What were the total scores for Alice in the CSV?"*
   - *"Summarize Sheet1 of the Excel document."*
   - *"Explain the methodology described on page 3 of the PDF."*
