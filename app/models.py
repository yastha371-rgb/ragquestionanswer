from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class DocumentUploadRequest(BaseModel):
    """Request model for document upload metadata."""
    title: str = Field(..., min_length=1, max_length=255, description="Document title")
    description: Optional[str] = Field(None, max_length=1000, description="Optional document summary/description")


class DocumentInfo(BaseModel):
    """Document information response model."""
    doc_id: str = Field(..., description="Unique document identifier")
    title: str = Field(..., description="Document title")
    description: Optional[str] = Field(None, description="Document description")
    file_name: str = Field(..., description="Original file name")
    uploaded_at: datetime = Field(default_factory=datetime.now, description="Upload timestamp")
    chunk_count: int = Field(0, description="Number of text chunks indexed")
    size_bytes: int = Field(0, description="File size in bytes")


class Question(BaseModel):
    """Question request model for RAG pipeline."""
    query: str = Field(..., min_length=3, max_length=2000, description="Academic inquiry or question")
    conversation_id: Optional[str] = Field(None, description="Optional multi-turn session ID")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of context chunks to retrieve")


class SourceReference(BaseModel):
    """Reference to source document chunk used in answer generation."""
    doc_id: str = Field(..., description="Source document ID")
    title: str = Field(..., description="Source document title")
    chunk_text: str = Field(..., description="Retrieved excerpt text")
    relevance_score: float = Field(..., description="Cosine similarity score (0.0 to 1.0)")
    page_number: Optional[int] = Field(None, description="Source page number if available")


class AnswerResponse(BaseModel):
    """Answer response model with source citations and confidence scoring."""
    answer: str = Field(..., description="Synthesized academic response")
    sources: List[SourceReference] = Field(default_factory=list, description="Source citations")
    confidence_score: float = Field(..., description="Aggregate retrieval confidence (0.0 to 1.0)")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
    query: str = Field(..., description="Original user query")
    conversation_id: Optional[str] = Field(None, description="Conversation session ID")
    cached: bool = Field(False, description="Whether response was served from cache")


class ConversationMessage(BaseModel):
    """Individual message in a conversation thread."""
    role: str = Field(..., description="Message role ('user' or 'assistant')")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp")
    sources: Optional[List[SourceReference]] = Field(None, description="Source citations for assistant replies")


class ConversationHistory(BaseModel):
    """Conversation history model holding session messages."""
    conversation_id: str = Field(..., description="Session identifier")
    messages: List[ConversationMessage] = Field(default_factory=list, description="Ordered conversation messages")
    created_at: datetime = Field(default_factory=datetime.now, description="Session creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")


class HealthCheckResponse(BaseModel):
    """Health check response schema."""
    status: str
    version: str
    services: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)
