import os
from typing import List, Tuple
from app.config import settings


def validate_file_extension(filename: str, allowed_types: List[str]) -> bool:
    """Check if the uploaded file's extension is in the permitted list."""
    if not filename or "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[-1].lower()
    return ext in [t.lower().lstrip(".") for t in allowed_types]


def sanitize_query(query: str) -> str:
    """Sanitize and clean query string from leading/trailing whitespaces and control chars."""
    if not query:
        return ""
    # Strip whitespace and normalize spaces
    cleaned = " ".join(query.strip().split())
    return cleaned


def validate_file_size(size_bytes: int, max_bytes: int = settings.MAX_FILE_SIZE) -> Tuple[bool, str]:
    """Validate that file size is within limits and not empty."""
    if size_bytes <= 0:
        return False, "File is empty (0 bytes)."
    if size_bytes > max_bytes:
        max_mb = max_bytes / (1024 * 1024)
        curr_mb = size_bytes / (1024 * 1024)
        return False, f"File size ({curr_mb:.2f}MB) exceeds maximum limit of {max_mb:.2f}MB."
    return True, ""
