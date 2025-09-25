"""
Caching Module for Modern RAG Application

This module provides caching mechanisms for query results to improve performance
by avoiding redundant processing of repeated queries.
"""

import os
import time
import json
import hashlib
import logging
from typing import Dict, Any, Optional, Tuple, List
from functools import lru_cache
from pathlib import Path
import asyncio
import pickle

from pydantic import Field
from pydantic_settings import BaseSettings
from langchain.docstore.document import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CacheConfig(BaseSettings):
    """Configuration settings for caching operations."""
    cache_dir: str = Field("./cache", env="CACHE_DIR")
    cache_ttl: int = Field(3600, env="CACHE_TTL")  # Time-to-live in seconds
    max_cache_size: int = Field(100, env="MAX_CACHE_SIZE")  # Maximum number of cached items
    enable_disk_cache: bool = Field(True, env="ENABLE_DISK_CACHE")
    enable_memory_cache: bool = Field(True, env="ENABLE_MEMORY_CACHE")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


@lru_cache()
def get_cache_config() -> CacheConfig:
    """Get the cache configuration."""
    return CacheConfig()


class QueryCache:
    """Cache for query results to improve performance."""
    
    def __init__(self):
        """Initialize the query cache."""
        self.config = get_cache_config()
        self._memory_cache: Dict[str, Tuple[Any, float]] = {}  # {key: (value, timestamp)}
        self._ensure_cache_dir()
    
    def _ensure_cache_dir(self):
        """Ensure the cache directory exists."""
        if self.config.enable_disk_cache:
            os.makedirs(self.config.cache_dir, exist_ok=True)
            logger.info(f"Cache directory ensured at: {self.config.cache_dir}")
    
    def _generate_key(self, query: str, **kwargs) -> str:
        """Generate a cache key from the query and additional parameters.
        
        Args:
            query: The query string
            **kwargs: Additional parameters that affect the result
            
        Returns:
            A unique hash key for the query and parameters
        """
        # Create a string representation of the query and parameters
        key_parts = [query]
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
        key_string = "|".join(key_parts)
        
        # Generate a hash of the string
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _is_expired(self, timestamp: float) -> bool:
        """Check if a cached item has expired.
        
        Args:
            timestamp: The timestamp when the item was cached
            
        Returns:
            True if the item has expired, False otherwise
        """
        return time.time() - timestamp > self.config.cache_ttl
    
    def _clean_memory_cache(self):
        """Remove expired items from the memory cache."""
        if not self.config.enable_memory_cache:
            return
            
        # Remove expired items
        expired_keys = [
            key for key, (_, timestamp) in self._memory_cache.items()
            if self._is_expired(timestamp)
        ]
        for key in expired_keys:
            del self._memory_cache[key]
        
        # If still too many items, remove oldest
        if len(self._memory_cache) > self.config.max_cache_size:
            sorted_items = sorted(
                self._memory_cache.items(),
                key=lambda x: x[1][1]  # Sort by timestamp
            )
            # Keep only the newest items
            to_keep = sorted_items[-self.config.max_cache_size:]
            self._memory_cache = {k: v for k, v in to_keep}
    
    def _get_disk_cache_path(self, key: str) -> Path:
        """Get the file path for a disk cache item.
        
        Args:
            key: The cache key
            
        Returns:
            Path to the cache file
        """
        return Path(self.config.cache_dir) / f"{key}.pickle"
    
    async def get(self, query: str, **kwargs) -> Optional[Any]:
        """Get a cached result for a query.
        
        Args:
            query: The query string
            **kwargs: Additional parameters that affect the result
            
        Returns:
            The cached result, or None if not found or expired
        """
        key = self._generate_key(query, **kwargs)
        
        # Check memory cache first
        if self.config.enable_memory_cache and key in self._memory_cache:
            value, timestamp = self._memory_cache[key]
            if not self._is_expired(timestamp):
                logger.info(f"Cache hit (memory): {query[:50]}...")
                return value
        
        # Check disk cache if enabled
        if self.config.enable_disk_cache:
            cache_path = self._get_disk_cache_path(key)
            if cache_path.exists():
                try:
                    # Load from disk asynchronously
                    content = await asyncio.to_thread(
                        self._load_from_disk, cache_path
                    )
                    if content:
                        value, timestamp = content
                        if not self._is_expired(timestamp):
                            # Update memory cache
                            if self.config.enable_memory_cache:
                                self._memory_cache[key] = (value, timestamp)
                            logger.info(f"Cache hit (disk): {query[:50]}...")
                            return value
                except Exception as e:
                    logger.error(f"Error loading cache from disk: {str(e)}")
        
        logger.info(f"Cache miss: {query[:50]}...")
        return None
    
    def _load_from_disk(self, path: Path) -> Optional[Tuple[Any, float]]:
        """Load a cached item from disk.
        
        Args:
            path: Path to the cache file
            
        Returns:
            Tuple of (value, timestamp) or None if loading fails
        """
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load cache from {path}: {str(e)}")
            return None
    
    async def set(self, query: str, value: Any, **kwargs):
        """Cache a result for a query.
        
        Args:
            query: The query string
            value: The result to cache
            **kwargs: Additional parameters that affect the result
        """
        key = self._generate_key(query, **kwargs)
        timestamp = time.time()
        
        # Update memory cache
        if self.config.enable_memory_cache:
            self._memory_cache[key] = (value, timestamp)
            self._clean_memory_cache()
        
        # Update disk cache if enabled
        if self.config.enable_disk_cache:
            cache_path = self._get_disk_cache_path(key)
            try:
                # Save to disk asynchronously
                await asyncio.to_thread(
                    self._save_to_disk, cache_path, (value, timestamp)
                )
            except Exception as e:
                logger.error(f"Error saving cache to disk: {str(e)}")
        
        logger.info(f"Cached result for query: {query[:50]}...")
    
    def _save_to_disk(self, path: Path, content: Tuple[Any, float]):
        """Save a cached item to disk.
        
        Args:
            path: Path to the cache file
            content: Tuple of (value, timestamp) to save
        """
        try:
            with open(path, 'wb') as f:
                pickle.dump(content, f)
        except Exception as e:
            logger.error(f"Failed to save cache to {path}: {str(e)}")
    
    async def clear(self):
        """Clear all cached items."""
        # Clear memory cache
        if self.config.enable_memory_cache:
            self._memory_cache = {}
        
        # Clear disk cache if enabled
        if self.config.enable_disk_cache:
            try:
                cache_dir = Path(self.config.cache_dir)
                if cache_dir.exists():
                    for cache_file in cache_dir.glob("*.pickle"):
                        await asyncio.to_thread(os.remove, cache_file)
            except Exception as e:
                logger.error(f"Error clearing disk cache: {str(e)}")
        
        logger.info("Cache cleared")
    
    async def clear_expired(self):
        """Clear expired cached items."""
        # Clear expired items from memory cache
        if self.config.enable_memory_cache:
            self._clean_memory_cache()
        
        # Clear expired items from disk cache if enabled
        if self.config.enable_disk_cache:
            try:
                cache_dir = Path(self.config.cache_dir)
                if cache_dir.exists():
                    for cache_file in cache_dir.glob("*.pickle"):
                        try:
                            content = await asyncio.to_thread(
                                self._load_from_disk, cache_file
                            )
                            if content:
                                _, timestamp = content
                                if self._is_expired(timestamp):
                                    await asyncio.to_thread(os.remove, cache_file)
                        except Exception as e:
                            logger.error(f"Error processing cache file {cache_file}: {str(e)}")
            except Exception as e:
                logger.error(f"Error clearing expired disk cache: {str(e)}")
        
        logger.info("Expired cache items cleared")


# Create a singleton instance
query_cache = QueryCache()


# Async API functions
async def get_cached_result(query: str, **kwargs) -> Optional[Any]:
    """Get a cached result for a query."""
    return await query_cache.get(query, **kwargs)


async def cache_result(query: str, value: Any, **kwargs):
    """Cache a result for a query."""
    await query_cache.set(query, value, **kwargs)


async def clear_cache():
    """Clear all cached items."""
    await query_cache.clear()


async def clear_expired_cache():
    """Clear expired cached items."""
    await query_cache.clear_expired()
