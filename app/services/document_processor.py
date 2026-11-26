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
    def extract_pages_from_pdf(file_path: str) -> List[Dict[str, Any]]:
        """Extract individual pages with metadata from PDF for precise attribution."""
        pages = []
        try:
            reader = PdfReader(file_path)
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages.append({
                        "page_number": page_num + 1,
                        "text": page_text
                    })
        except Exception as e:
            raise ValueError(f"Failed to extract pages from PDF: {str(e)}")
        return pages

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
            
            all_text = "\n".join(paragraphs + table_texts)
            return all_text
        except ImportError:
            raise ValueError("python-docx is not installed. Please install it to process .docx files.")
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
            
            # Convert rows into readable entity records
            for idx, row in df.iterrows():
                row_items = []
                for col in df.columns:
                    val = row[col]
                    if pd.notna(val):
                        row_items.append(f"{col}: {val}")
                if row_items:
                    lines.append(f"Row {idx + 1}: " + " | ".join(row_items))
            
            return "\n".join(lines)
        except Exception as e:
            # Fallback to standard csv reader if pandas fails
            try:
                lines = [f"[CSV File: {Path(file_path).name}]"]
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    reader = csv.reader(f)
                    headers = next(reader, None)
                    if headers:
                        lines.append(f"Columns: {', '.join(headers)}\n")
                    for idx, row in enumerate(reader):
                        if headers and len(row) == len(headers):
                            row_str = " | ".join([f"{h}: {v.strip()}" for h, v in zip(headers, row) if v.strip()])
                        else:
                            row_str = " | ".join([v.strip() for v in row if v.strip()])
                        if row_str:
                            lines.append(f"Row {idx + 1}: {row_str}")
                return "\n".join(lines)
            except Exception as fallback_err:
                raise ValueError(f"Failed to parse CSV file: {str(e)} (Fallback error: {str(fallback_err)})")

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
                        row_items = []
                        for col in df.columns:
                            val = row[col]
                            if pd.notna(val):
                                row_items.append(f"{col}: {val}")
                        if row_items:
                            output_parts.append(f"Row {row_idx + 1}: " + " | ".join(row_items))
                    output_parts.append("")
                
            return "\n".join(output_parts)
        except Exception as e:
            raise ValueError(f"Failed to extract text from Excel file: {str(e)}")

    @staticmethod
    def process_document(file_path: str, file_type: str) -> str:
        """
        Process document and extract text based on file format.
        
        Args:
            file_path: Path to the file
            file_type: Extension (pdf, txt, docx, csv, xlsx, xls)
            
        Returns:
            Extracted text content formatted for semantic search & vector indexing
        """
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
            raise ValueError(
                f"Unsupported file type: .{file_type}. "
                f"Supported types are: {', '.join(settings.ALLOWED_FILE_TYPES)}"
            )

    @staticmethod
    def validate_file(file_path: str, max_size: int = settings.MAX_FILE_SIZE) -> Tuple[bool, str]:
        """
        Validate uploaded file existence and size.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not os.path.exists(file_path):
            return False, "File not found on server."
        
        size = os.path.getsize(file_path)
        return validate_file_size(size, max_size)
