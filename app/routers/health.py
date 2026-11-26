from fastapi import APIRouter
from datetime import datetime
from app.config import settings
from app.models import HealthCheckResponse
from app.services.cache_service import CacheService
from app.services.retrieval_service import RetrievalService

router = APIRouter(prefix="/health", tags=["Health"])

cache_service = CacheService()
retrieval_service = RetrievalService()


@router.get("", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    """Comprehensive system health and dependent service status check."""
    redis_connected = cache_service.is_connected()
    
    # Check ChromaDB
    try:
        doc_count = retrieval_service.get_count()
        chroma_status = "healthy"
    except Exception as e:
        doc_count = 0
        chroma_status = f"unhealthy: {str(e)}"

    gemini_configured = bool(settings.api_key and not settings.api_key.startswith("dummy"))
    overall_status = "healthy" if (chroma_status == "healthy") else "degraded"

    return HealthCheckResponse(
        status=overall_status,
        version=settings.API_VERSION,
        services={
            "api": "healthy",
            "chroma_db": {
                "status": chroma_status,
                "total_vectors": doc_count,
                "storage_path": settings.CHROMA_PATH
            },
            "redis_cache": {
                "status": "connected" if redis_connected else "disconnected",
                "host": settings.REDIS_HOST,
                "port": settings.REDIS_PORT
            },
            "gemini_ai_configuration": {
                "model": settings.MODEL_NAME,
                "embedding_model": settings.EMBEDDING_MODEL,
                "api_key_set": gemini_configured,
                "provider": "Google Gemini (Free Tier)"
            }
        },
        timestamp=datetime.now()
    )


@router.get("/live")
async def liveness_probe() -> dict:
    """Basic liveness probe for orchestrators."""
    return {"status": "alive"}
