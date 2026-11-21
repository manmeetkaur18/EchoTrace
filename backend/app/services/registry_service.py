import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

# Configure module logger for server-side diagnostics
logger = logging.getLogger(__name__)

# Dynamic path resolution anchored to the project structure:
# __file__ -> backend/app/services/registry_service.py
# .parent -> services/
# .parent.parent -> app/
# .parent.parent.parent -> backend/
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_REGISTRY_DB_DIR = BACKEND_DIR / "data"
DEFAULT_REGISTRY_DB_PATH = DEFAULT_REGISTRY_DB_DIR / "registry.sqlite3"


# ---------------------------------------------------------
# Custom Exception Hierarchy
# ---------------------------------------------------------
class RegistryError(Exception):
    """Base exception for active-version registry operational failures."""
    pass


class RegistryUnavailableError(RegistryError):
    """Raised when the SQLite registry database is unreachable, locked, or corrupted."""
    pass


class StaleVersionError(RegistryError):
    """Raised when a concurrent ingestion job attempts to activate a stale version (CAS failure)."""
    pass


# ---------------------------------------------------------
# In-Process Thread-Safe Recording Lock Manager
# ---------------------------------------------------------
_recording_locks: Dict[str, threading.Lock] = {}
_recording_locks_mutex = threading.Lock()


@contextmanager
def acquire_recording_lock(recording_id: str, timeout: float = 15.0) -> Generator[None, None, None]:
    """
    Acquires an in-process lock scoped strictly to the given recording_id.
    Ensures different recordings can be processed concurrently while preventing race conditions
    between simultaneous ingestions and cleanups of the exact same recording within the process.

    Args:
        recording_id (str): Recording identifier to lock.
        timeout (float): Maximum seconds to wait to acquire the lock.

    Raises:
        TimeoutError: If the lock cannot be acquired within the specified timeout.
    """
    clean_id = str(recording_id).strip()
    with _recording_locks_mutex:
        if clean_id not in _recording_locks:
            _recording_locks[clean_id] = threading.Lock()
        lock = _recording_locks[clean_id]

    acquired = lock.acquire(timeout=timeout)
    if not acquired:
        logger.error(
            "Timed out after %.1fs waiting for recording lock on '%s'.",
            timeout,
            clean_id,
        )
        raise TimeoutError(
            f"Could not acquire lock for recording '{clean_id}' within {timeout}s."
        )

    try:
        yield
    finally:
        try:
            lock.release()
        except RuntimeError:
            pass


# ---------------------------------------------------------
# SQLite Connection Management
# ---------------------------------------------------------
def _resolve_db_path(db_path: Optional[Union[str, Path]] = None) -> Path:
    """Resolves and ensures the directory for the target SQLite database exists."""
    if db_path is not None:
        p = Path(db_path).resolve()
    else:
        p = DEFAULT_REGISTRY_DB_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


@contextmanager
def get_db_connection(db_path: Optional[Union[str, Path]] = None) -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager yielding a configured SQLite connection with WAL mode, busy timeout,
    and foreign key constraints enabled.

    Args:
        db_path (Optional[Union[str, Path]]): Path to SQLite file (defaults to backend/data/registry.sqlite3).

    Yields:
        sqlite3.Connection: Active database connection.

    Raises:
        RegistryUnavailableError: If connection or execution fails due to database errors.
    """
    resolved = _resolve_db_path(db_path)
    conn: Optional[sqlite3.Connection] = None
    try:
        conn = sqlite3.connect(
            str(resolved),
            timeout=10.0,
            check_same_thread=False,
            isolation_level=None,  # Autocommit mode by default; manual transactions managed via BEGIN/COMMIT
        )
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for high concurrency between readers and writers
        conn.execute("PRAGMA journal_mode = WAL;")
        # Set busy timeout (10 seconds) to tolerate short write locks
        conn.execute("PRAGMA busy_timeout = 10000;")
        conn.execute("PRAGMA foreign_keys = ON;")
        yield conn
    except sqlite3.Error as sqle:
        logger.error("SQLite operational error on database '%s': %s", resolved, sqle)
        raise RegistryUnavailableError(f"Registry database error on '{resolved}': {sqle}") from sqle
    except Exception as exc:
        if isinstance(exc, (RegistryError, TimeoutError)):
            raise
        logger.error("Unexpected error accessing registry at '%s': %s", resolved, exc)
        raise RegistryUnavailableError(f"Failed to access registry database at '{resolved}': {exc}") from exc
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


# ---------------------------------------------------------
# Schema Initialization
# ---------------------------------------------------------
def init_registry(db_path: Optional[Union[str, Path]] = None) -> None:
    """
    Initializes the SQLite active-version registry schema safely and idempotently.

    Tables:
    - recording_versions:
        Holds the single active authoritative version for each registered recording,
        along with an incremental generation number for optimistic concurrency (CAS).
    - ingestion_versions:
        Audit log of all staged, active, superseded, or failed ingestion attempts,
        including created_at and updated_at timestamps for heartbeat tracking.
    """
    with get_db_connection(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE;")
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS recording_versions (
                    recording_id TEXT PRIMARY KEY,
                    active_version TEXT NOT NULL,
                    generation INTEGER NOT NULL DEFAULT 1,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ingestion_versions (
                    recording_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    status TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    activated_at REAL,
                    PRIMARY KEY (recording_id, version)
                );
                """
            )

            # Migration guard: add updated_at column to ingestion_versions if missing
            cursor = conn.execute("PRAGMA table_info(ingestion_versions);")
            cols = {row["name"] for row in cursor.fetchall()}
            if cols and "updated_at" not in cols:
                conn.execute("ALTER TABLE ingestion_versions ADD COLUMN updated_at REAL;")
                conn.execute("UPDATE ingestion_versions SET updated_at = created_at WHERE updated_at IS NULL;")

            conn.execute("COMMIT;")
            logger.info("Registry schema verified/initialized successfully.")
        except Exception:
            conn.execute("ROLLBACK;")
            raise


# ---------------------------------------------------------
# Read Operations
# ---------------------------------------------------------
def get_active_version(
    recording_id: str,
    db_path: Optional[Union[str, Path]] = None,
) -> Optional[str]:
    """
    Reads the active version for a given recording ID.

    Distinguishes:
    1. Legacy recording: returns None (no row in recording_versions).
    2. Versioned recording: returns active_version string.
    3. Registry unavailable or corrupted: raises RegistryUnavailableError.

    Never treats registry errors as legacy recordings!
    """
    if isinstance(recording_id, bool) or not isinstance(recording_id, str):
        raise TypeError(f"recording_id must be a string, got {type(recording_id).__name__}.")
    clean_id = recording_id.strip()
    if not clean_id:
        raise ValueError("recording_id must be a non-empty string.")

    with get_db_connection(db_path) as conn:
        cursor = conn.execute(
            "SELECT active_version FROM recording_versions WHERE recording_id = ?;",
            (clean_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return str(row["active_version"])


def get_recording_generation(
    recording_id: str,
    db_path: Optional[Union[str, Path]] = None,
) -> int:
    """
    Retrieves the current generation number for a recording.
    Returns 0 if the recording has no registry entry yet (legacy or brand new).
    """
    if isinstance(recording_id, bool) or not isinstance(recording_id, str):
        raise TypeError(f"recording_id must be a string, got {type(recording_id).__name__}.")
    clean_id = recording_id.strip()
    if not clean_id:
        raise ValueError("recording_id must be a non-empty string.")

    with get_db_connection(db_path) as conn:
        cursor = conn.execute(
            "SELECT generation FROM recording_versions WHERE recording_id = ?;",
            (clean_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return 0
        return int(row["generation"])


def get_version_statuses(
    recording_id: str,
    db_path: Optional[Union[str, Path]] = None,
) -> Dict[str, str]:
    """
    Returns a mapping of {version: status} for all registered versions of a recording
    from ingestion_versions.

    Statuses: 'staged', 'active', 'superseded', 'failed'.
    """
    if isinstance(recording_id, bool) or not isinstance(recording_id, str):
        raise TypeError(f"recording_id must be a string, got {type(recording_id).__name__}.")
    clean_id = recording_id.strip()
    if not clean_id:
        raise ValueError("recording_id must be a non-empty string.")

    with get_db_connection(db_path) as conn:
        cursor = conn.execute(
            "SELECT version, status FROM ingestion_versions WHERE recording_id = ?;",
            (clean_id,),
        )
        return {str(row["version"]): str(row["status"]) for row in cursor.fetchall()}


def list_active_versions(
    db_path: Optional[Union[str, Path]] = None,
) -> Dict[str, str]:
    """
    Returns a mapping of {recording_id: active_version} for all currently versioned recordings.
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.execute(
            "SELECT recording_id, active_version FROM recording_versions;"
        )
        return {str(row["recording_id"]): str(row["active_version"]) for row in cursor.fetchall()}


def list_inactive_versions(
    recording_id: str,
    db_path: Optional[Union[str, Path]] = None,
) -> List[str]:
    """
    Lists all version strings for a recording that are NOT currently active
    (e.g., 'superseded', 'staged', or 'failed').
    """
    clean_id = str(recording_id).strip()
    with get_db_connection(db_path) as conn:
        cursor = conn.execute(
            """
            SELECT version FROM ingestion_versions
            WHERE recording_id = ? AND status != 'active';
            """,
            (clean_id,),
        )
        return [str(row["version"]) for row in cursor.fetchall()]


# ---------------------------------------------------------
# Version Lifecycle & Concurrency (Stage, Heartbeat & Activate)
# ---------------------------------------------------------
def stage_version(
    recording_id: str,
    version: str,
    expected_generation: int,
    db_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Registers a new version in 'staged' status before validating chunks and writing to ChromaDB.
    Sets both created_at and updated_at to the current timestamp.
    """
    clean_rec_id = str(recording_id).strip()
    clean_version = str(version).strip()
    now = time.time()

    with get_db_connection(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE;")
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO ingestion_versions
                (recording_id, version, status, generation, created_at, updated_at, activated_at)
                VALUES (?, ?, 'staged', ?, ?, ?, NULL);
                """,
                (clean_rec_id, clean_version, expected_generation, now, now),
            )
            conn.execute("COMMIT;")
            logger.info(
                "Staged version '%s' for recording '%s' (generation %d).",
                clean_version,
                clean_rec_id,
                expected_generation,
            )
        except Exception:
            conn.execute("ROLLBACK;")
            raise


def update_version_heartbeat(
    recording_id: str,
    version: str,
    db_path: Optional[Union[str, Path]] = None,
) -> bool:
    """
    Updates the updated_at heartbeat timestamp of a currently 'staged' version.
    Ensures long-running ingestions do not get considered stale by recovery.

    Returns:
        bool: True if a staged version row was updated, False if the version is no longer staged.

    Raises:
        RegistryUnavailableError: If SQLite connection or execution fails.
    """
    clean_rec_id = str(recording_id).strip()
    clean_version = str(version).strip()
    now = time.time()

    with get_db_connection(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE;")
        try:
            cursor = conn.execute(
                """
                UPDATE ingestion_versions
                SET updated_at = ?
                WHERE recording_id = ? AND version = ? AND status = 'staged';
                """,
                (now, clean_rec_id, clean_version),
            )
            conn.execute("COMMIT;")
            return cursor.rowcount > 0
        except Exception:
            conn.execute("ROLLBACK;")
            raise


def activate_version(
    recording_id: str,
    version: str,
    expected_generation: int,
    db_path: Optional[Union[str, Path]] = None,
) -> int:
    """
    Atomically activates a completed and verified ingestion version in SQLite.

    Guarantees:
    - Atomically updates recording_versions.
    - Compares generation: if current generation != expected_generation, raises StaleVersionError (CAS failure).
    - Increments generation by 1.
    - Fencing: verifies target ingestion version exists and has status 'staged'.
    - Conditionally transitions target version to 'active'. If rowcount != 1, raises StaleVersionError.
    - Marks previous active version as 'superseded' in ingestion_versions.
    - Any failure rolls back both table updates atomically.
    - Never uses INSERT OR REPLACE to reactivate an existing failed or superseded version.

    Returns:
        int: The new generation integer.

    Raises:
        StaleVersionError: If concurrent activation altered generation or target version is not staged.
        RegistryUnavailableError: If SQLite write fails.
    """
    clean_rec_id = str(recording_id).strip()
    clean_version = str(version).strip()
    now = time.time()

    with get_db_connection(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE;")
        try:
            cursor = conn.execute(
                "SELECT active_version, generation FROM recording_versions WHERE recording_id = ?;",
                (clean_rec_id,),
            )
            existing = cursor.fetchone()

            if existing is not None:
                current_gen = int(existing["generation"])
                if current_gen != expected_generation:
                    raise StaleVersionError(
                        f"Generation mismatch for recording '{clean_rec_id}': "
                        f"expected {expected_generation}, found {current_gen}. "
                        f"A newer ingestion has already been activated; rejecting stale activation of version '{clean_version}'."
                    )
                new_gen = current_gen + 1
            else:
                # First versioned ingestion for this recording
                if expected_generation != 0:
                    raise StaleVersionError(
                        f"Expected generation {expected_generation} for recording '{clean_rec_id}', but no record exists."
                    )
                new_gen = 1

            # 1. Activation fencing: Verify target version is 'staged' and update to 'active'
            ingest_cursor = conn.execute(
                """
                UPDATE ingestion_versions
                SET status = 'active', generation = ?, updated_at = ?, activated_at = ?
                WHERE recording_id = ? AND version = ? AND status = 'staged';
                """,
                (new_gen, now, now, clean_rec_id, clean_version),
            )
            if ingest_cursor.rowcount != 1:
                raise StaleVersionError(
                    f"Cannot activate version '{clean_version}' for recording '{clean_rec_id}': "
                    f"version is not in staged status (lease expired or version marked failed)."
                )

            # 2. Update recording_versions with generation CAS check
            if existing is not None:
                cursor = conn.execute(
                    """
                    UPDATE recording_versions
                    SET active_version = ?, generation = ?, updated_at = ?
                    WHERE recording_id = ? AND generation = ?;
                    """,
                    (clean_version, new_gen, now, clean_rec_id, current_gen),
                )
                if cursor.rowcount != 1:
                    raise StaleVersionError(
                        f"Failed compare-and-swap update for recording '{clean_rec_id}' (generation changed concurrently)."
                    )

                # Mark prior active version as superseded (excluding newly activated clean_version)
                conn.execute(
                    """
                    UPDATE ingestion_versions
                    SET status = 'superseded', updated_at = ?
                    WHERE recording_id = ? AND status = 'active' AND version != ?;
                    """,
                    (now, clean_rec_id, clean_version),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO recording_versions
                    (recording_id, active_version, generation, created_at, updated_at)
                    VALUES (?, ?, 1, ?, ?);
                    """,
                    (clean_rec_id, clean_version, now, now),
                )

            conn.execute("COMMIT;")
            logger.info(
                "Successfully activated version '%s' for recording '%s' (generation -> %d).",
                clean_version,
                clean_rec_id,
                new_gen,
            )
            return new_gen

        except Exception:
            conn.execute("ROLLBACK;")
            raise


def mark_version_failed(
    recording_id: str,
    version: str,
    db_path: Optional[Union[str, Path]] = None,
) -> bool:
    """
    Marks a version as 'failed' in the audit table.

    Safety:
    - Only updates rows where status = 'staged'.
    - NEVER modifies or downgrades an 'active' or 'superseded' version to 'failed'.

    Returns:
        bool: True if a staged row was transitioned to 'failed', False otherwise.
    """
    clean_rec_id = str(recording_id).strip()
    clean_version = str(version).strip()
    now = time.time()

    try:
        with get_db_connection(db_path) as conn:
            conn.execute("BEGIN IMMEDIATE;")
            try:
                cursor = conn.execute(
                    """
                    UPDATE ingestion_versions
                    SET status = 'failed', updated_at = ?
                    WHERE recording_id = ? AND version = ? AND status = 'staged';
                    """,
                    (now, clean_rec_id, clean_version),
                )
                conn.execute("COMMIT;")
                updated = cursor.rowcount > 0
                if updated:
                    logger.info(
                        "Marked staged version '%s' as failed for recording '%s'.",
                        clean_version,
                        clean_rec_id,
                    )
                else:
                    logger.info(
                        "Version '%s' for recording '%s' was not in 'staged' status; status unchanged.",
                        clean_version,
                        clean_rec_id,
                    )
                return updated
            except Exception:
                conn.execute("ROLLBACK;")
                raise
    except Exception as exc:
        logger.warning(
            "Could not mark version '%s' as failed for recording '%s': %s",
            clean_version,
            clean_rec_id,
            exc,
        )
        return False


def recover_stale_staged_versions(
    recording_id: Optional[str] = None,
    timeout_seconds: float = 300.0,
    db_path: Optional[Union[str, Path]] = None,
) -> List[str]:
    """
    Conservative recovery mechanism for abandoned staged versions.

    Rules:
    1. Scoped to a specific recording_id if provided (acquires recording lock).
    2. When recording_id is None (global recovery), queries candidate staged versions,
       acquires acquire_recording_lock(candidate_recording_id) individually for each candidate,
       rechecks version status and heartbeat cutoff inside the lock, and executes the
       conditional transition inside a short SQLite transaction.
    3. Only inspects versions where status = 'staged'.
    4. Checks the heartbeat lease: (now - COALESCE(updated_at, created_at)) > timeout_seconds.
    5. Re-checks the status during the atomic update.
    6. NEVER touches 'active' or 'superseded' versions.
    7. Returns list of recovered version IDs.

    Args:
        recording_id (Optional[str]): Optional recording ID to scope recovery.
        timeout_seconds (float): Minimum seconds since last heartbeat to consider abandoned (default 300s).
        db_path (Optional[Union[str, Path]]): Path to SQLite database.

    Returns:
        List[str]: List of version IDs transitioned from 'staged' to 'failed'.
    """
    clean_rec_id = recording_id.strip() if recording_id and isinstance(recording_id, str) else None
    effective_timeout = max(timeout_seconds, 1.0)

    if clean_rec_id:
        with acquire_recording_lock(clean_rec_id):
            recovered_versions: List[str] = []
            now = time.time()
            cutoff_time = now - effective_timeout
            with get_db_connection(db_path) as conn:
                conn.execute("BEGIN IMMEDIATE;")
                try:
                    cursor = conn.execute(
                        """
                        SELECT version, COALESCE(updated_at, created_at) as last_seen
                        FROM ingestion_versions
                        WHERE recording_id = ? AND status = 'staged';
                        """,
                        (clean_rec_id,),
                    )
                    candidates = cursor.fetchall()
                    for cand in candidates:
                        v_id = str(cand["version"])
                        last_seen = float(cand["last_seen"]) if cand["last_seen"] is not None else 0.0
                        if last_seen <= cutoff_time:
                            update_cursor = conn.execute(
                                """
                                UPDATE ingestion_versions
                                SET status = 'failed', updated_at = ?
                                WHERE recording_id = ? AND version = ? AND status = 'staged'
                                  AND COALESCE(updated_at, created_at) <= ?;
                                """,
                                (now, clean_rec_id, v_id, cutoff_time),
                            )
                            if update_cursor.rowcount > 0:
                                recovered_versions.append(v_id)
                                logger.info(
                                    "Recovered abandoned staged version '%s' for recording '%s' (idle %.1fs).",
                                    v_id,
                                    clean_rec_id,
                                    now - last_seen,
                                )
                    conn.execute("COMMIT;")
                except Exception:
                    conn.execute("ROLLBACK;")
                    raise
            return recovered_versions
    else:
        # Global recovery: query candidates first without holding write transactions
        now = time.time()
        cutoff_time = now - effective_timeout
        candidate_items: List[tuple[str, str]] = []
        with get_db_connection(db_path) as conn:
            cursor = conn.execute(
                """
                SELECT recording_id, version
                FROM ingestion_versions
                WHERE status = 'staged' AND COALESCE(updated_at, created_at) <= ?;
                """,
                (cutoff_time,),
            )
            for row in cursor.fetchall():
                candidate_items.append((str(row["recording_id"]), str(row["version"])))

        recovered_versions: List[str] = []
        for cand_rec_id, cand_version in candidate_items:
            try:
                with acquire_recording_lock(cand_rec_id):
                    cand_now = time.time()
                    cand_cutoff = cand_now - effective_timeout
                    with get_db_connection(db_path) as conn:
                        conn.execute("BEGIN IMMEDIATE;")
                        try:
                            # Recheck status and cutoff inside lock
                            check_cursor = conn.execute(
                                """
                                SELECT COALESCE(updated_at, created_at) as last_seen
                                FROM ingestion_versions
                                WHERE recording_id = ? AND version = ? AND status = 'staged';
                                """,
                                (cand_rec_id, cand_version),
                            )
                            row = check_cursor.fetchone()
                            if row is not None:
                                last_seen = float(row["last_seen"]) if row["last_seen"] is not None else 0.0
                                if last_seen <= cand_cutoff:
                                    update_cursor = conn.execute(
                                        """
                                        UPDATE ingestion_versions
                                        SET status = 'failed', updated_at = ?
                                        WHERE recording_id = ? AND version = ? AND status = 'staged'
                                          AND COALESCE(updated_at, created_at) <= ?;
                                        """,
                                        (cand_now, cand_rec_id, cand_version, cand_cutoff),
                                    )
                                    if update_cursor.rowcount > 0:
                                        recovered_versions.append(cand_version)
                                        logger.info(
                                            "Global recovery: recovered abandoned staged version '%s' for recording '%s' (idle %.1fs).",
                                            cand_version,
                                            cand_rec_id,
                                            cand_now - last_seen,
                                        )
                            conn.execute("COMMIT;")
                        except Exception:
                            conn.execute("ROLLBACK;")
                            raise
            except TimeoutError:
                logger.warning(
                    "Global recovery skipped recording '%s' version '%s': could not acquire recording lock.",
                    cand_rec_id,
                    cand_version,
                )
                continue

        return recovered_versions
