import time
import uuid
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

from fastapi import APIRouter, HTTPException
from app.config import settings
from app.models import (
    Question,
    AnswerResponse,
    SourceReference,
    ConversationHistory,
    ConversationMessage,
)
from app.services.cache_service import CacheService
from app.services.embedding_service import EmbeddingService
from app.services.generation_service import GenerationService
from app.services.retrieval_service import RetrievalService
from app.utils.validators import sanitize_query

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/qa", tags=["Q&A"])

embedding_service = EmbeddingService()
retrieval_service = RetrievalService()
generation_service = GenerationService()
cache_service = CacheService()

# In-memory conversation session registry
conversations: Dict[str, Dict[str, Any]] = {}


@router.post("/ask", response_model=AnswerResponse)
async def ask_question(question: Question) -> AnswerResponse:
    """
    Submit an academic question to the RAG pipeline.
    Checks Redis cache, performs vector retrieval, verifies confidence, and generates cited answer.
    """
    start_time = time.time()
    clean_query = sanitize_query(question.query)

    if not clean_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        # 1. Cache hit inspection
        cached_response = cache_service.get(clean_query, question.conversation_id)
        if cached_response:
            cached_response["cached"] = True
            return AnswerResponse(**cached_response)

        # 2. Vector embedding of inquiry
        query_embedding = embedding_service.get_embedding(clean_query)

        # 3. Semantic retrieval
        top_k = question.top_k or settings.RETRIEVAL_TOP_K
        retrieved_docs = retrieval_service.retrieve(query_embedding, top_k=top_k)

        if not retrieved_docs:
            raise HTTPException(
                status_code=404,
                detail="No relevant academic literature or documents found in the knowledge base."
            )

        # 4. Context aggregation with document metadata and page indicators
        context_blocks = []
        for doc in retrieved_docs:
            meta = doc.get("metadata", {})
            title = meta.get("title", meta.get("doc_id", "Unknown Document"))
            page = meta.get("page_number")
            page_info = f", Page {page}" if page else ""
            context_blocks.append(f"[Source: {title}{page_info}]\n{doc['text']}")

        context_str = "\n\n".join(context_blocks)

        # 5. Answer synthesis via LLM
        answer, _ = generation_service.generate_answer(clean_query, context_str)

        # 6. Confidence evaluation
        confidence_score = generation_service.calculate_confidence_score(retrieved_docs)

        if confidence_score < settings.MIN_CONFIDENCE_SCORE:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Retrieval confidence score ({confidence_score:.2f}) fell below "
                    f"the required enterprise threshold of {settings.MIN_CONFIDENCE_SCORE}."
                )
            )

        # 7. Construct source citations
        sources: List[SourceReference] = []
        for doc in retrieved_docs[:3]:
            meta = doc.get("metadata", {})
            text_preview = doc["text"][:400] + "..." if len(doc["text"]) > 400 else doc["text"]
            sources.append(
                SourceReference(
                    doc_id=str(meta.get("doc_id", "unknown")),
                    title=str(meta.get("title", "Academic Document")),
                    chunk_text=text_preview,
                    relevance_score=doc["relevance_score"],
                    page_number=meta.get("page_number")
                )
            )

        total_time_ms = round((time.time() - start_time) * 1000, 2)

        response = AnswerResponse(
            answer=answer,
            sources=sources,
            confidence_score=confidence_score,
            processing_time_ms=total_time_ms,
            query=clean_query,
            conversation_id=question.conversation_id,
            cached=False
        )

        # 8. Cache response for future queries
        # Convert response to dictionary compatible with JSON serialization
        response_dict = response.model_dump()
        cache_service.set(clean_query, response_dict, question.conversation_id)

        # 9. Multi-turn conversation tracking
        if question.conversation_id:
            if question.conversation_id not in conversations:
                conversations[question.conversation_id] = {
                    "conversation_id": question.conversation_id,
                    "created_at": datetime.now(),
                    "messages": []
                }
            
            conversations[question.conversation_id]["messages"].append(
                ConversationMessage(
                    role="user",
                    content=clean_query,
                    timestamp=datetime.now()
                )
            )
            conversations[question.conversation_id]["messages"].append(
                ConversationMessage(
                    role="assistant",
                    content=answer,
                    timestamp=datetime.now(),
                    sources=sources
                )
            )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing Q&A query: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during query processing: {str(e)}"
        )


@router.post("/conversation/start")
async def start_conversation() -> dict:
    """Initialize a new multi-turn conversation session."""
    conversation_id = str(uuid.uuid4())
    conversations[conversation_id] = {
        "conversation_id": conversation_id,
        "created_at": datetime.now(),
        "messages": []
    }
    return {
        "conversation_id": conversation_id,
        "created_at": datetime.now().isoformat(),
        "status": "active"
    }


@router.get("/conversation/{conversation_id}", response_model=ConversationHistory)
async def get_conversation(conversation_id: str) -> ConversationHistory:
    """Retrieve full conversation transcript and source attributions for a given session."""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation session not found.")

    conv = conversations[conversation_id]
    return ConversationHistory(
        conversation_id=conversation_id,
        messages=conv["messages"],
        created_at=conv["created_at"],
        updated_at=datetime.now()
    )


@router.delete("/conversation/{conversation_id}")
async def clear_conversation(conversation_id: str) -> dict:
    """Clear conversation history and associated session cache."""
    if conversation_id in conversations:
        del conversations[conversation_id]
    cache_service.clear_conversation(conversation_id)
    return {"message": f"Conversation session {conversation_id} cleared."}


@router.get("/cache/stats")
async def get_cache_stats() -> dict:
    """Retrieve Redis cache health and performance statistics."""
    return cache_service.get_stats()
