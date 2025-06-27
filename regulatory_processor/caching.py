"""
Caching module for regulatory document processing to improve performance and reliability.
"""

import os
import json
import pickle
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import sqlite3
from dataclasses import dataclass, asdict
import threading

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""
    key: str
    data: Any
    created_at: datetime
    expires_at: Optional[datetime]
    file_hash: str
    file_size: int
    access_count: int = 0
    last_accessed: Optional[datetime] = None


class DocumentCache:
    """
    Advanced caching system for regulatory document processing.
    
    Features:
    - File-based caching with SQLite metadata
    - Automatic cache invalidation based on file changes
    - TTL (Time To Live) support
    - Memory and disk cache management
    - Thread-safe operations
    """
    
    def __init__(self, cache_dir: str = ".regulatory_cache", max_memory_size: int = 100):
        """
        Initialize the document cache.
        
        Args:
            cache_dir: Directory to store cache files
            max_memory_size: Maximum number of entries to keep in memory
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.db_path = self.cache_dir / "cache_metadata.db"
        self.data_dir = self.cache_dir / "data"
        self.data_dir.mkdir(exist_ok=True)
        
        self.max_memory_size = max_memory_size
        self.memory_cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        
        self._init_database()
        self._cleanup_expired()
        
        logger.info(f"DocumentCache initialized with cache_dir: {cache_dir}")
    
    def _init_database(self):
        """Initialize SQLite database for cache metadata."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    file_path TEXT,
                    file_hash TEXT,
                    file_size INTEGER,
                    created_at TEXT,
                    expires_at TEXT,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TEXT,
                    data_type TEXT,
                    success BOOLEAN DEFAULT TRUE
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires_at ON cache_entries(expires_at)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_file_hash ON cache_entries(file_hash)
            """)
    
    def _generate_key(self, file_path: str, operation: str, **kwargs) -> str:
        """Generate a unique cache key."""
        # Include operation parameters in key generation
        params_str = json.dumps(sorted(kwargs.items()), default=str)
        combined = f"{file_path}:{operation}:{params_str}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def _get_file_hash(self, file_path: str) -> str:
        """Generate hash of file content for cache invalidation."""
        if not os.path.exists(file_path):
            return ""
        
        hash_obj = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_obj.update(chunk)
            return hash_obj.hexdigest()
        except Exception as e:
            logger.warning(f"Failed to hash file {file_path}: {e}")
            return ""
    
    def _get_data_file_path(self, key: str) -> Path:
        """Get the path for storing cached data."""
        return self.data_dir / f"{key}.pkl"
    
    def _save_to_disk(self, key: str, data: Any) -> bool:
        """Save data to disk cache."""
        try:
            data_file = self._get_data_file_path(key)
            with open(data_file, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            return True
        except Exception as e:
            logger.error(f"Failed to save cache data to disk: {e}")
            return False
    
    def _load_from_disk(self, key: str) -> Optional[Any]:
        """Load data from disk cache."""
        try:
            data_file = self._get_data_file_path(key)
            if data_file.exists():
                with open(data_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load cache data from disk: {e}")
        return None
    
    def put(self, file_path: str, operation: str, data: Any, 
            ttl_hours: Optional[int] = 24, **kwargs) -> str:
        """
        Store data in cache.
        
        Args:
            file_path: Path to the source file
            operation: Type of operation (e.g., 'extraction', 'validation')
            data: Data to cache
            ttl_hours: Time to live in hours (None for no expiration)
            **kwargs: Additional parameters that affect caching
            
        Returns:
            Cache key
        """
        with self._lock:
            key = self._generate_key(file_path, operation, **kwargs)
            
            # Get file metadata
            file_hash = self._get_file_hash(file_path)
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            
            # Calculate expiration
            created_at = datetime.now()
            expires_at = created_at + timedelta(hours=ttl_hours) if ttl_hours else None
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                data=data,
                created_at=created_at,
                expires_at=expires_at,
                file_hash=file_hash,
                file_size=file_size
            )
            
            # Save to disk
            if self._save_to_disk(key, data):
                # Store in memory cache (manage size)
                if len(self.memory_cache) >= self.max_memory_size:
                    self._evict_memory_cache()
                self.memory_cache[key] = entry
                
                # Store metadata in database
                self._store_metadata(key, file_path, entry, operation)
                
                logger.debug(f"Cached {operation} result for {file_path} with key {key[:8]}...")
                return key
            else:
                logger.error(f"Failed to cache {operation} result for {file_path}")
                return ""
    
    def get(self, file_path: str, operation: str, **kwargs) -> Optional[Any]:
        """
        Retrieve data from cache.
        
        Args:
            file_path: Path to the source file
            operation: Type of operation
            **kwargs: Additional parameters
            
        Returns:
            Cached data or None if not found/expired/invalid
        """
        with self._lock:
            key = self._generate_key(file_path, operation, **kwargs)
            
            # Check if file has changed
            if not self._is_cache_valid(key, file_path):
                self._remove_entry(key)
                return None
            
            # Try memory cache first
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                if self._is_entry_valid(entry):
                    entry.access_count += 1
                    entry.last_accessed = datetime.now()
                    self._update_access_metadata(key, entry)
                    logger.debug(f"Cache hit (memory) for {operation} on {file_path}")
                    return entry.data
                else:
                    # Expired entry
                    self._remove_entry(key)
                    return None
            
            # Try disk cache
            data = self._load_from_disk(key)
            if data is not None:
                # Load metadata and check validity
                entry_meta = self._get_entry_metadata(key)
                if entry_meta and self._is_entry_valid_from_meta(entry_meta):
                    # Add back to memory cache
                    entry = CacheEntry(
                        key=key,
                        data=data,
                        created_at=datetime.fromisoformat(entry_meta['created_at']),
                        expires_at=datetime.fromisoformat(entry_meta['expires_at']) if entry_meta['expires_at'] else None,
                        file_hash=entry_meta['file_hash'],
                        file_size=entry_meta['file_size'],
                        access_count=entry_meta['access_count'] + 1,
                        last_accessed=datetime.now()
                    )
                    
                    if len(self.memory_cache) >= self.max_memory_size:
                        self._evict_memory_cache()
                    self.memory_cache[key] = entry
                    
                    self._update_access_metadata(key, entry)
                    logger.debug(f"Cache hit (disk) for {operation} on {file_path}")
                    return data
                else:
                    # Invalid or expired
                    self._remove_entry(key)
            
            logger.debug(f"Cache miss for {operation} on {file_path}")
            return None
    
    def _is_cache_valid(self, key: str, file_path: str) -> bool:
        """Check if cache entry is still valid based on file changes."""
        if not os.path.exists(file_path):
            return False
        
        entry_meta = self._get_entry_metadata(key)
        if not entry_meta:
            return False
        
        current_hash = self._get_file_hash(file_path)
        return current_hash == entry_meta['file_hash']
    
    def _is_entry_valid(self, entry: CacheEntry) -> bool:
        """Check if cache entry is still valid (not expired)."""
        if entry.expires_at is None:
            return True
        return datetime.now() < entry.expires_at
    
    def _is_entry_valid_from_meta(self, meta: Dict[str, Any]) -> bool:
        """Check if cache entry is valid from metadata."""
        if not meta['expires_at']:
            return True
        expires_at = datetime.fromisoformat(meta['expires_at'])
        return datetime.now() < expires_at
    
    def _store_metadata(self, key: str, file_path: str, entry: CacheEntry, operation: str):
        """Store cache entry metadata in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO cache_entries 
                    (key, file_path, file_hash, file_size, created_at, expires_at, 
                     access_count, last_accessed, data_type, success)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    key, file_path, entry.file_hash, entry.file_size,
                    entry.created_at.isoformat(),
                    entry.expires_at.isoformat() if entry.expires_at else None,
                    entry.access_count,
                    entry.last_accessed.isoformat() if entry.last_accessed else None,
                    operation, True
                ))
        except Exception as e:
            logger.error(f"Failed to store cache metadata: {e}")
    
    def _get_entry_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cache entry metadata from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM cache_entries WHERE key = ?", (key,)
                )
                row = cursor.fetchone()
                return dict(row) if row else None
        except Exception as e:
            logger.error(f"Failed to get cache metadata: {e}")
            return None
    
    def _update_access_metadata(self, key: str, entry: CacheEntry):
        """Update access statistics in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE cache_entries 
                    SET access_count = ?, last_accessed = ?
                    WHERE key = ?
                """, (entry.access_count, entry.last_accessed.isoformat(), key))
        except Exception as e:
            logger.error(f"Failed to update access metadata: {e}")
    
    def _remove_entry(self, key: str):
        """Remove cache entry from memory, disk, and database."""
        # Remove from memory
        self.memory_cache.pop(key, None)
        
        # Remove from disk
        data_file = self._get_data_file_path(key)
        if data_file.exists():
            try:
                data_file.unlink()
            except Exception as e:
                logger.error(f"Failed to remove cache file: {e}")
        
        # Remove from database
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
        except Exception as e:
            logger.error(f"Failed to remove cache metadata: {e}")
    
    def _evict_memory_cache(self):
        """Evict least recently used entries from memory cache."""
        if not self.memory_cache:
            return
        
        # Sort by last accessed time (least recent first)
        sorted_entries = sorted(
            self.memory_cache.items(),
            key=lambda x: x[1].last_accessed or x[1].created_at
        )
        
        # Remove oldest 25% of entries
        num_to_remove = max(1, len(sorted_entries) // 4)
        for key, _ in sorted_entries[:num_to_remove]:
            del self.memory_cache[key]
        
        logger.debug(f"Evicted {num_to_remove} entries from memory cache")
    
    def _cleanup_expired(self):
        """Remove expired cache entries."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Find expired entries
                now = datetime.now().isoformat()
                cursor = conn.execute(
                    "SELECT key FROM cache_entries WHERE expires_at IS NOT NULL AND expires_at < ?",
                    (now,)
                )
                expired_keys = [row[0] for row in cursor.fetchall()]
                
                # Remove expired entries
                for key in expired_keys:
                    self._remove_entry(key)
                
                if expired_keys:
                    logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
        except Exception as e:
            logger.error(f"Failed to cleanup expired entries: {e}")
    
    def clear(self, operation_type: Optional[str] = None):
        """Clear cache entries, optionally filtered by operation type."""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    if operation_type:
                        cursor = conn.execute(
                            "SELECT key FROM cache_entries WHERE data_type = ?",
                            (operation_type,)
                        )
                    else:
                        cursor = conn.execute("SELECT key FROM cache_entries")
                    
                    keys_to_remove = [row[0] for row in cursor.fetchall()]
                    
                    for key in keys_to_remove:
                        self._remove_entry(key)
                    
                    logger.info(f"Cleared {len(keys_to_remove)} cache entries" + 
                              (f" for operation type: {operation_type}" if operation_type else ""))
            except Exception as e:
                logger.error(f"Failed to clear cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_entries,
                        SUM(access_count) as total_accesses,
                        AVG(access_count) as avg_accesses,
                        COUNT(CASE WHEN expires_at IS NOT NULL AND expires_at < ? THEN 1 END) as expired_entries,
                        data_type,
                        COUNT(*) as count_by_type
                    FROM cache_entries 
                    GROUP BY data_type
                """, (datetime.now().isoformat(),))
                
                results = cursor.fetchall()
                
                return {
                    'memory_cache_size': len(self.memory_cache),
                    'total_disk_entries': sum(row[5] for row in results),
                    'total_accesses': sum(row[1] for row in results if row[1]),
                    'expired_entries': sum(row[3] for row in results if row[3]),
                    'cache_by_type': {row[4]: row[5] for row in results if row[4]},
                    'cache_directory': str(self.cache_dir),
                    'database_size': os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
                }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {'error': str(e)}


# Global cache instance
_global_cache: Optional[DocumentCache] = None


def get_cache(cache_dir: str = ".regulatory_cache") -> DocumentCache:
    """Get or create global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = DocumentCache(cache_dir)
    return _global_cache


def cache_result(operation: str, ttl_hours: int = 24, **cache_kwargs):
    """
    Decorator for caching function results based on file input.
    
    Args:
        operation: Operation name for cache key generation
        ttl_hours: Time to live in hours
        **cache_kwargs: Additional parameters for cache key generation
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Extract file path from arguments (assume first argument is file path)
            if args:
                file_path = str(args[0])
                cache = get_cache()
                
                # Try to get from cache
                cache_key_params = {**cache_kwargs, **kwargs}
                cached_result = cache.get(file_path, operation, **cache_key_params)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                try:
                    result = func(*args, **kwargs)
                    cache.put(file_path, operation, result, ttl_hours, **cache_key_params)
                    return result
                except Exception as e:
                    logger.error(f"Function {func.__name__} failed: {e}")
                    raise
            else:
                # No file path, execute without caching
                return func(*args, **kwargs)
        
        return wrapper
    return decorator