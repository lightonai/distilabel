import sqlite3
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from contextlib import contextmanager
from threading import Lock

class RoutedToCacheDB:
    """SQLite-based cache for 'routed_to' data."""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "routed_to.db"
        self._lock = Lock()
        self._logger = logging.getLogger("distilabel.routed_to_cache")
        self._signatures: Optional[set[str]] = None
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database with the routed_to cache table."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS routed_to (
                    signature TEXT PRIMARY KEY,
                    batch_routed_to TEXT NOT NULL
                )
            """)
            conn.commit()
        self._load_signatures_to_memory()

    def _load_signatures_to_memory(self) -> None:
        """Loads all signatures from the DB into an in-memory set for O(1) lookups."""
        self._logger.info("Loading 'routed_to' signatures into memory for faster lookups...")
        signatures = set()
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT signature FROM routed_to")
            for row in cursor:
                signatures.add(row[0])
        self._signatures = signatures

    @contextmanager
    def _get_connection(self):
        """Get a database connection with proper error handling."""
        conn = None
        try:
            conn = sqlite3.connect(
                self.db_path,
                timeout=30.0,
                isolation_level=None
            )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            yield conn
        finally:
            if conn:
                conn.close()

    def get(self, signature: str) -> Optional[Any]:
        """Retrieve a cached routed_to entry."""
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.execute(
                        "SELECT batch_routed_to FROM routed_to WHERE signature = ?",
                        (signature,)
                    )
                    row = cursor.fetchone()
                    if row is None:
                        return None
                    return json.loads(row[0])
            except Exception as e:
                self._logger.warning(f"Failed to read from routed_to cache: {e}")
                return None

    def set(self, signature: str, batch_routed_to: Any) -> None:
        """Store a routed_to entry in the cache."""
        with self._lock:
            try:
                with self._get_connection() as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO routed_to (signature, batch_routed_to)
                        VALUES (?, ?)
                    """, (
                        signature,
                        json.dumps(batch_routed_to),
                    ))
                self._signatures.add(signature)
            except Exception as e:
                self._logger.warning(f"Failed to write to routed_to cache: {e}")

    def exists(self, signature: str) -> bool:
        """Check if a routed_to cache entry exists."""
        return signature in self._signatures

# Global cache instance
_cache_instances: Dict[str, "RoutedToCacheDB"] = {}
_cache_lock = Lock()

def get_routed_to_cache_db(cache_dir: Path) -> "RoutedToCacheDB":
    """Get or create a cache instance for the given directory."""
    cache_key = str(cache_dir.resolve())

    # First check without a lock for performance
    if cache_key in _cache_instances:
        return _cache_instances[cache_key]

    # If not found, acquire lock and check again (double-checked locking)
    with _cache_lock:
        if cache_key not in _cache_instances:
            _cache_instances[cache_key] = RoutedToCacheDB(cache_dir)
        return _cache_instances[cache_key] 

