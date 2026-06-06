import psycopg
from psycopg import Connection
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector

from config import DATABASE_URL, EMBED_DIM
from logger import get_logger

logger = get_logger(__name__)


def get_db() -> Connection:
    conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)
    register_vector(conn)
    return conn


def init_db() -> None:
    logger.info("running init_db — ensuring schema exists")
    conn = get_db()
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id BIGSERIAL PRIMARY KEY,
            filename TEXT NOT NULL,
            filepath TEXT NOT NULL,
            pages INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id BIGSERIAL PRIMARY KEY,
            doc_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            chunk_text TEXT NOT NULL,
            chunk_index INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS vec_chunks (
            chunk_id BIGINT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
            embedding vector({EMBED_DIM}) NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id         BIGSERIAL PRIMARY KEY,
            session_id TEXT NOT NULL,
            turn_id    TEXT NOT NULL,
            role       TEXT NOT NULL,
            content    TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS vec_conversations (
            conversation_id BIGINT PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
            embedding       vector({EMBED_DIM}) NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()
    logger.info("schema ready (embed_dim=%d)", EMBED_DIM)
