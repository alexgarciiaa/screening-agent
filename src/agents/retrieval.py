"""Retrieval over the Supabase pgvector knowledge base (the RAG step).

Embeds the candidate's question with Voyage and asks Postgres for the closest
knowledge-base chunks via the ``match_kb_chunks`` function. Any failure degrades
gracefully to an empty result, so the bot still answers without the knowledge base.
"""

import logging
from dataclasses import dataclass

from sqlalchemy import text

from ..config import Settings
from ..storage.db import create_db_engine

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    source: str
    heading: str | None
    content: str
    similarity: float


def _to_vector_literal(embedding: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in embedding) + "]"


class Retriever:
    """Voyage query embedding + pgvector similarity search in Supabase."""

    def __init__(self, settings: Settings) -> None:
        import voyageai

        self._settings = settings
        self._client = voyageai.Client(api_key=settings.voyage_api_key)
        self._engine = create_db_engine(settings.database_url)

    def search(self, query: str) -> list[RetrievedChunk]:
        try:
            embedding = self._client.embed(
                [query], model=self._settings.embedding_model, input_type="query"
            ).embeddings[0]
            with self._engine.connect() as conn:
                rows = conn.execute(
                    text(
                        "select source, heading, content, similarity "
                        "from match_kb_chunks(cast(:emb as vector), :k, :thr)"
                    ),
                    {
                        "emb": _to_vector_literal(embedding),
                        "k": self._settings.rag_top_k,
                        "thr": self._settings.rag_min_similarity,
                    },
                ).all()
            return [RetrievedChunk(*row) for row in rows]
        except Exception:
            logger.warning("Knowledge-base retrieval failed; answering without it", exc_info=True)
            return []


def build_retriever(settings: Settings) -> Retriever | None:
    """Return a retriever when Voyage is configured, else None (RAG disabled)."""
    if not settings.voyage_api_key:
        return None
    return Retriever(settings)
