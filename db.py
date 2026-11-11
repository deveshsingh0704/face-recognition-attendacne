"""
Database module for SQLite schema and queries.

Handles:
- SQLite table creation (students, sessions, attendance)
- Embedding serialization (numpy array <-> binary BLOB)
- Student CRUD operations and uniqueness enforcement
- Session tracking and attendance logging with primary key constraints
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from config import DB_PATH


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Open an SQLite connection with Row factory enabled for column name access."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    # Enable foreign key support in SQLite
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: Path = DB_PATH) -> None:
    """Initialize SQLite tables if they do not exist."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        # 1. Students table: Stores roll number, name, and serialized 128-d embeddings
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                roll_no TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                embeddings BLOB NOT NULL,
                enrolled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )

        # 2. Sessions table: Stores individual lecture/class sessions
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                start_time TEXT NOT NULL,
                course TEXT NOT NULL
            );
            """
        )

        # 3. Attendance table: Stores attendance per student per session
        # Enforces UNIQUE(session_id, student_id) via composite PRIMARY KEY
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS attendance (
                session_id INTEGER NOT NULL,
                student_id INTEGER NOT NULL,
                status TEXT NOT NULL,
                confidence REAL NOT NULL,
                frames_seen INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                PRIMARY KEY (session_id, student_id),
                FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE,
                FOREIGN KEY (student_id) REFERENCES students (id) ON DELETE CASCADE
            );
            """
        )
        conn.commit()


def serialize_embeddings(embeddings: np.ndarray) -> bytes:
    """
    Serialize a (K, 128) or (128,) float32 numpy array into raw bytes.

    Args:
        embeddings: numpy float32 array.

    Returns:
        Raw bytes suitable for SQLite BLOB storage.
    """
    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != 128:
        raise ValueError(f"Expected 128-dimensional embedding, got shape {arr.shape}")
    return arr.tobytes()


def deserialize_embeddings(blob: bytes) -> np.ndarray:
    """
    Deserialize raw binary BLOB into a (K, 128) float32 numpy array.

    Args:
        blob: SQLite BLOB bytes.

    Returns:
        np.ndarray of shape (K, 128) with dtype float32.
    """
    if not blob:
        return np.empty((0, 128), dtype=np.float32)
    arr = np.frombuffer(blob, dtype=np.float32)
    return arr.reshape(-1, 128)


def insert_student(
    roll_no: str,
    name: str,
    embeddings: np.ndarray,
    db_path: Path = DB_PATH,
) -> int:
    """
    Insert a newly registered student into the database.

    Raises:
        sqlite3.IntegrityError: If roll_no already exists.
    """
    blob = serialize_embeddings(embeddings)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO students (roll_no, name, embeddings)
            VALUES (?, ?, ?);
            """,
            (roll_no.strip(), name.strip(), blob),
        )
        conn.commit()
        return cursor.lastrowid


def update_student_embeddings(
    roll_no: str,
    embeddings: np.ndarray,
    db_path: Path = DB_PATH,
) -> bool:
    """Update or append embeddings for an existing student."""
    blob = serialize_embeddings(embeddings)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE students
            SET embeddings = ?, enrolled_at = CURRENT_TIMESTAMP
            WHERE roll_no = ?;
            """,
            (blob, roll_no.strip()),
        )
        conn.commit()
        return cursor.rowcount > 0


def get_student_by_roll(roll_no: str, db_path: Path = DB_PATH) -> Optional[Dict[str, Any]]:
    """Retrieve a student record and parsed embeddings by roll number."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, roll_no, name, embeddings, enrolled_at FROM students WHERE roll_no = ?;",
            (roll_no.strip(),),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "roll_no": row["roll_no"],
            "name": row["name"],
            "embeddings": deserialize_embeddings(row["embeddings"]),
            "enrolled_at": row["enrolled_at"],
        }


def get_all_students(db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    """Retrieve all enrolled students with deserialized embeddings."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, roll_no, name, embeddings, enrolled_at FROM students ORDER BY roll_no ASC;"
        )
        rows = cursor.fetchall()
        students = []
        for row in rows:
            students.append(
                {
                    "id": row["id"],
                    "roll_no": row["roll_no"],
                    "name": row["name"],
                    "embeddings": deserialize_embeddings(row["embeddings"]),
                    "enrolled_at": row["enrolled_at"],
                }
            )
        return students


def delete_student(roll_no: str, db_path: Path = DB_PATH) -> bool:
    """Delete a student and cascade delete their attendance records."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM students WHERE roll_no = ?;", (roll_no.strip(),))
        conn.commit()
        return cursor.rowcount > 0


def create_session(
    course: str,
    date: Optional[str] = None,
    start_time: Optional[str] = None,
    db_path: Path = DB_PATH,
) -> int:
    """Create a new attendance session record and return session_id."""
    now = datetime.now()
    session_date = date or now.strftime("%Y-%m-%d")
    session_time = start_time or now.strftime("%H:%M:%S")

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO sessions (date, start_time, course)
            VALUES (?, ?, ?);
            """,
            (session_date, session_time, course.strip()),
        )
        conn.commit()
        return cursor.lastrowid


def record_attendance(
    session_id: int,
    student_id: int,
    status: str,
    confidence: float,
    frames_seen: int,
    timestamp: Optional[str] = None,
    db_path: Path = DB_PATH,
) -> None:
    """
    Record attendance for a student in a session.
    Uses INSERT OR REPLACE to update if record already exists.
    """
    ts = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO attendance (session_id, student_id, status, confidence, frames_seen, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id, student_id) DO UPDATE SET
                status = excluded.status,
                confidence = excluded.confidence,
                frames_seen = excluded.frames_seen,
                timestamp = excluded.timestamp;
            """,
            (session_id, student_id, status.upper(), float(confidence), int(frames_seen), ts),
        )
        conn.commit()


def get_session_attendance(session_id: int, db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    """Retrieve full attendance list for a specific session joined with student names."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT 
                s.roll_no,
                s.name,
                a.session_id,
                a.student_id,
                a.status,
                a.confidence,
                a.frames_seen,
                a.timestamp
            FROM attendance a
            JOIN students s ON a.student_id = s.id
            WHERE a.session_id = ?
            ORDER BY s.roll_no ASC;
            """,
            (session_id,),
        )
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


def get_all_sessions(db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    """List all recorded sessions."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, date, start_time, course FROM sessions ORDER BY id DESC;"
        )
        return [dict(r) for r in cursor.fetchall()]
