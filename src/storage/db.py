from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


def _normalize_db_url(url: str) -> str:
    """Route the plain Postgres URI that Supabase gives through psycopg 3."""
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://") :]
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://") :]
    return url


def create_db_engine(database_url: str):
    url = _normalize_db_url(database_url)
    # Disable prepared statements on Postgres so it works behind Supabase's
    # connection poolers (transaction mode rejects them).
    connect_args = {"prepare_threshold": None} if url.startswith("postgresql") else {}
    return create_engine(url, future=True, connect_args=connect_args)
