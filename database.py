"""
Database
--------
PostgreSQL operations for the yt-agent2 project.
"""
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()


# Each function below opens and closes its own connection by design.
# This is intentional for a single-user app where concurrent requests are rare.
# If this app ever becomes multi-user, replace with a connection pool
# (e.g. psycopg2.pool.SimpleConnectionPool) to avoid per-request connection overhead.
def get_connection():
    """Open and return a psycopg2 connection using DATABASE_URL."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise EnvironmentError(
            "DATABASE_URL not found. Add it to your environment variables."
        )

    try:
        conn = psycopg2.connect(database_url)
        print("[Database] Connected successfully")
        return conn
    except Exception as e:
        print(f"[Database] Connection failed: {e}")
        raise


def init_db():
    """Create the videos table if it does not already exist."""
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS videos (
                    video_id   TEXT PRIMARY KEY,
                    url        TEXT NOT NULL,
                    title      TEXT,
                    transcript TEXT,
                    blog_post  TEXT,
                    logs       TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
        conn.commit()
        print("[Database] videos table ready")
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"[Database] init_db failed: {e}")
        raise
    finally:
        if conn:
            conn.close()


def get_video_by_id(video_id: str):
    """Return a video row as a dict, or None if not found."""
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM videos WHERE video_id = %s",
                (video_id,),
            )
            row = cur.fetchone()

        if row:
            print(f"[Database] Found video: {video_id}")
            return dict(row)

        print(f"[Database] Video not found: {video_id}")
        return None
    except Exception as e:
        print(f"[Database] get_video_by_id failed: {e}")
        raise
    finally:
        if conn:
            conn.close()


def save_video(
    video_id: str,
    url: str,
    title: str,
    transcript: str,
    blog_post: str,
    logs: str,
):
    """Insert a new video row. Does nothing if video_id already exists."""
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO videos (video_id, url, title, transcript, blog_post, logs)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (video_id) DO NOTHING
                """,
                (video_id, url, title, transcript, blog_post, logs),
            )
            inserted = cur.rowcount > 0
        conn.commit()

        if inserted:
            print(f"[Database] Saved video: {video_id}")
        else:
            print(f"[Database] Video already exists, skipped: {video_id}")
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"[Database] save_video failed: {e}")
        raise
    finally:
        if conn:
            conn.close()


def update_blog_post(video_id: str, blog_post: str):
    """Update the blog_post column for an existing video."""
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE videos SET blog_post = %s WHERE video_id = %s",
                (blog_post, video_id),
            )
            updated = cur.rowcount > 0
        conn.commit()

        if updated:
            print(f"[Database] Updated blog_post for video: {video_id}")
        else:
            print(f"[Database] Video not found for update: {video_id}")
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"[Database] update_blog_post failed: {e}")
        raise
    finally:
        if conn:
            conn.close()
