import sqlite3
import json
import logging
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Union
from contextlib import contextmanager
from threading import Lock
import time
from datetime import datetime, date

from distilabel.typing import GenerateOutput
from distilabel import utils


class LMCacheDB:
    """SQLite-based cache for language model responses.
    
    Uses a single SQLite database file instead of individual JSON files
    for better performance and reduced filesystem overhead.
    """
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "lm_cache.db"
        self._lock = Lock()
        self._logger = logging.getLogger("distilabel.lm_cache")
        
        # Initialize database
        self._init_db()
    
    def _init_db(self):
        """Initialize the SQLite database with the cache table."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS lm_cache (
                    cache_key TEXT PRIMARY KEY,
                    response_data TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    last_accessed REAL NOT NULL,
                    model_name TEXT
                )
            """)
            
            # Create indexes for faster lookups
            conn.execute("CREATE INDEX IF NOT EXISTS idx_model_name ON lm_cache(model_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON lm_cache(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_last_accessed ON lm_cache(last_accessed)")
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get a database connection with proper error handling."""
        conn = None
        try:
            conn = sqlite3.connect(
                self.db_path,
                timeout=30.0,  # 30 second timeout
                isolation_level=None  # autocommit mode
            )
            # Enable WAL mode for better concurrent access
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            yield conn
        finally:
            if conn:
                conn.close()
    
    def _create_cache_key(self, cache_params: Dict[str, Any]) -> str:
        """Create a cache key from parameters."""
        return utils.hash_structure_with_images(cache_params)
    
    def get(self, cache_params: Dict[str, Any]) -> Optional[GenerateOutput]:
        """Retrieve a cached response."""
        cache_key = self._create_cache_key(cache_params)
        
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.execute(
                        "SELECT response_data FROM lm_cache WHERE cache_key = ?",
                        (cache_key,)
                    )
                    row = cursor.fetchone()
                    
                    if row is None:
                        return None
                    
                    # Update last accessed time
                    conn.execute(
                        "UPDATE lm_cache SET last_accessed = ? WHERE cache_key = ?",
                        (time.time(), cache_key)
                    )
                    
                    # Deserialize the response
                    response_data = json.loads(row[0])
                    return response_data
                    
            except Exception as e:
                self._logger.warning(f"Failed to read from cache: {e}")
                return None
    
    def set(self, cache_params: Dict[str, Any], response: GenerateOutput) -> None:
        """Store a response in the cache."""
        cache_key = self._create_cache_key(cache_params)
        model_name = cache_params.get('model_name', 'unknown')
        
        with self._lock:
            try:
                with self._get_connection() as conn:
                    current_time = time.time()
                    conn.execute("""
                        INSERT OR REPLACE INTO lm_cache 
                        (cache_key, response_data, created_at, last_accessed, model_name)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        cache_key,
                        json.dumps(response),
                        current_time,
                        current_time,
                        model_name,
                    ))
                    
            except Exception as e:
                self._logger.warning(f"Failed to write to cache: {e}")
    
    def exists(self, cache_params: Dict[str, Any]) -> bool:
        """Check if a cache entry exists."""
        cache_key = self._create_cache_key(cache_params)
        
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT 1 FROM lm_cache WHERE cache_key = ? LIMIT 1",
                    (cache_key,)
                )
                return cursor.fetchone() is not None
        except Exception:
            return False
    
    def clear_model_cache(self, model_name: str) -> int:
        """Clear all cache entries for a specific model."""
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.execute(
                        "DELETE FROM lm_cache WHERE model_name = ?",
                        (model_name,)
                    )
                    return cursor.rowcount
            except Exception as e:
                self._logger.warning(f"Failed to clear model cache: {e}")
                return 0
    
    def clear_old_entries(self, max_age_days: int = 30) -> int:
        """Clear cache entries older than max_age_days."""
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.execute(
                        "DELETE FROM lm_cache WHERE created_at < ?",
                        (cutoff_time,)
                    )
                    return cursor.rowcount
            except Exception as e:
                self._logger.warning(f"Failed to clear old entries: {e}")
                return 0
    
    def clear_date_range(
        self, 
        start_date: Optional[Union[str, date, datetime]] = None,
        end_date: Optional[Union[str, date, datetime]] = None
    ) -> int:
        """Clear cache entries within a date range.
        
        Args:
            start_date: Start date (inclusive). Can be string in YYYY-MM-DD format, 
                       date object, or datetime object. If None, no start limit.
            end_date: End date (inclusive). Can be string in YYYY-MM-DD format,
                     date object, or datetime object. If None, no end limit.
                     
        Returns:
            Number of entries cleared.
        """
        def parse_date(date_input: Union[str, date, datetime]) -> float:
            """Convert date input to timestamp."""
            if isinstance(date_input, str):
                # Parse YYYY-MM-DD format
                try:
                    parsed_date = datetime.strptime(date_input, "%Y-%m-%d")
                except ValueError:
                    raise ValueError(f"Date string must be in YYYY-MM-DD format, got: {date_input}")
            elif isinstance(date_input, date) and not isinstance(date_input, datetime):
                # Convert date to datetime at start of day
                parsed_date = datetime.combine(date_input, datetime.min.time())
            elif isinstance(date_input, datetime):
                parsed_date = date_input
            else:
                raise ValueError(f"Invalid date type: {type(date_input)}")
            
            return parsed_date.timestamp()
        
        conditions = []
        params = []
        
        if start_date is not None:
            start_timestamp = parse_date(start_date)
            conditions.append("created_at >= ?")
            params.append(start_timestamp)
        
        if end_date is not None:
            # Add 24 hours to end_date to make it inclusive of the entire day
            if isinstance(end_date, str):
                end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
                end_datetime = end_datetime.replace(hour=23, minute=59, second=59)
            elif isinstance(end_date, date) and not isinstance(end_date, datetime):
                end_datetime = datetime.combine(end_date, datetime.max.time())
            else:
                end_datetime = end_date
            
            end_timestamp = end_datetime.timestamp()
            conditions.append("created_at <= ?")
            params.append(end_timestamp)
        
        if not conditions:
            raise ValueError("At least one of start_date or end_date must be provided")
        
        query = f"DELETE FROM lm_cache WHERE {' AND '.join(conditions)}"
        
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.execute(query, params)
                    return cursor.rowcount
            except Exception as e:
                self._logger.warning(f"Failed to clear entries in date range: {e}")
                return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            with self._get_connection() as conn:
                # Total entries
                cursor = conn.execute("SELECT COUNT(*) FROM lm_cache")
                total_entries = cursor.fetchone()[0]
                
                # Entries by model
                cursor = conn.execute("""
                    SELECT model_name, COUNT(*) 
                    FROM lm_cache 
                    GROUP BY model_name
                """)
                model_counts = dict(cursor.fetchall())
                
                # Database size
                db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
                
                return {
                    'total_entries': total_entries,
                    'model_counts': model_counts,
                    'db_size_bytes': db_size,
                    'db_size_mb': round(db_size / (1024 * 1024), 2),
                    'db_path': str(self.db_path)
                }
                
        except Exception as e:
            self._logger.warning(f"Failed to get cache stats: {e}")
            return {}
    
    def vacuum(self) -> None:
        """Optimize the database by running VACUUM."""
        with self._lock:
            try:
                with self._get_connection() as conn:
                    conn.execute("VACUUM")
                self._logger.info("Database vacuum completed")
            except Exception as e:
                self._logger.warning(f"Failed to vacuum database: {e}")


# Global cache instance
_cache_instances: Dict[str, LMCacheDB] = {}
_cache_lock = Lock()


def get_lm_cache(cache_dir: Path) -> LMCacheDB:
    """Get or create a cache instance for the given directory."""
    cache_key = str(cache_dir.resolve())
    
    with _cache_lock:
        if cache_key not in _cache_instances:
            _cache_instances[cache_key] = LMCacheDB(cache_dir)
        return _cache_instances[cache_key]

