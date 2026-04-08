"""
Simple disk-based cache for expensive operations.
Thread-safe using file system atomicity.
"""
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class DiskCache:
    """Simple disk-based cache with JSON serialization."""
    
    def __init__(self, cache_dir: str = "./cache", max_age_seconds: Optional[int] = None):
        """
        Initialize disk cache.
        
        Args:
            cache_dir: Directory to store cache files
            max_age_seconds: Optional TTL for cache entries (None = no expiration)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age_seconds = max_age_seconds
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for a key."""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.json"
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key (will be hashed)
            
        Returns:
            Cached value or None if not found/expired
        """
        cache_file = self._get_cache_path(key)
        
        if not cache_file.exists():
            return None
        
        try:
            # Check age if max_age_seconds is set
            if self.max_age_seconds is not None:
                import time
                file_age = time.time() - cache_file.stat().st_mtime
                if file_age > self.max_age_seconds:
                    logger.debug(f"Cache expired for key: {key[:50]}...")
                    cache_file.unlink()  # Delete expired cache
                    return None
            
            # Load and return cached value
            with cache_file.open("r") as f:
                data = json.load(f)
                logger.debug(f"Cache hit for key: {key[:50]}...")
                return data
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to read cache for key {key[:50]}...: {e}")
            return None
    
    def set(self, key: str, value: Any) -> None:
        """
        Store value in cache.
        
        Args:
            key: Cache key (will be hashed)
            value: Value to cache (must be JSON-serializable)
        """
        cache_file = self._get_cache_path(key)
        
        try:
            # Write to temp file first, then atomic rename
            temp_file = cache_file.with_suffix(".tmp")
            with temp_file.open("w") as f:
                json.dump(value, f)
            temp_file.replace(cache_file)  # Atomic on POSIX systems
            logger.debug(f"Cache set for key: {key[:50]}...")
        except (TypeError, IOError) as e:
            logger.warning(f"Failed to write cache for key {key[:50]}...: {e}")
    
    def delete(self, key: str) -> None:
        """Delete cache entry for a key."""
        cache_file = self._get_cache_path(key)
        if cache_file.exists():
            cache_file.unlink()
            logger.debug(f"Cache deleted for key: {key[:50]}...")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
        logger.info("Cache cleared")
