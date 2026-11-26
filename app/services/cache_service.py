import json
import hashlib
import logging
from typing import Optional, Dict, Any
import redis
from app.config import settings

logger = logging.getLogger(__name__)


class CacheService:
    """Service for caching Q&A responses using Redis with graceful degradation."""

    def __init__(self):
        self._is_enabled = False
        try:
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                decode_responses=True,
                socket_connect_timeout=0.2,
                socket_timeout=0.2
            )
            self._is_enabled = bool(self.redis_client.ping())
        except Exception:
            self._is_enabled = False

    def is_connected(self) -> bool:
        """Check if Redis connection is active."""
        try:
            return bool(self.redis_client.ping())
        except Exception:
            return False

    def _generate_cache_key(self, query: str, conversation_id: Optional[str] = None) -> str:
        """Generate a deterministic cache key based on query hash and session."""
        normalized_query = " ".join(query.strip().lower().split())
        query_hash = hashlib.sha256(normalized_query.encode("utf-8")).hexdigest()

        if conversation_id:
            return f"academic_qa:conv:{conversation_id}:{query_hash}"
        return f"academic_qa:general:{query_hash}"

    def get(self, query: str, conversation_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get cached answer response if available.
        """
        if not self._is_enabled:
            return None

        try:
            cache_key = self._generate_cache_key(query, conversation_id)
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"Cache lookup failed (gracefully bypassing): {e}")
        return None

    def set(self, query: str, response: Dict[str, Any], conversation_id: Optional[str] = None) -> None:
        """
        Store an answer response in Redis with TTL.
        """
        if not self._is_enabled:
            return

        try:
            cache_key = self._generate_cache_key(query, conversation_id)
            self.redis_client.setex(
                cache_key,
                settings.CACHE_TTL,
                json.dumps(response, default=str)
            )
        except Exception as e:
            logger.warning(f"Cache write failed (gracefully continuing): {e}")

    def clear_conversation(self, conversation_id: str) -> None:
        """Clear all cached answers associated with a conversation session."""
        try:
            pattern = f"academic_qa:conv:{conversation_id}:*"
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
        except Exception as e:
            logger.warning(f"Cache invalidation failed: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Retrieve Redis cache operational statistics."""
        try:
            info = self.redis_client.info()
            return {
                "status": "connected",
                "used_memory": info.get("used_memory_human", "N/A"),
                "total_connections": info.get("total_connections_received", 0),
                "total_commands": info.get("total_commands_processed", 0),
                "uptime_days": round(info.get("uptime_in_seconds", 0) / 86400, 2)
            }
        except Exception as e:
            return {
                "status": "disconnected",
                "error": str(e)
            }
