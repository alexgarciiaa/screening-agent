"""Offline ingestion of the knowledge base into Supabase (pgvector).

Reads ``knowledge_base/*.md``, splits each document into sections, embeds them
with Voyage and upserts into ``kb_chunks``. Idempotent: a chunk is keyed by the
hash of its text, so only new or changed chunks are embedded, and chunks that no
longer exist in the docs are removed.

Run it whenever the documents change (needs VOYAGE_API_KEY and DATABASE_URL
pointing at Supabase):

    python -m scripts.ingest_kb
"""

import hashlib
import re
from pathlib import Path

from sqlalchemy import text

from src.config import get_settings
from src.storage.db import create_db_engine

KB_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"
_SECTION = re.compile(r"^##\s+(.*)$", re.MULTILINE)
_EMBED_BATCH = 128


def _title(markdown: str) -> str:
    for line in markdown.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def _sections(markdown: str) -> list[tuple[str, str]]:
    """Split a document into (heading, body) sections on '##' headings.

    Text before the first '##' (the intro) is attached to the document title.
    """
    title = _title(markdown)
    body = re.sub(r"^#\s+.*$", "", markdown, count=1, flags=re.MULTILINE).strip()
    parts = _SECTION.split(body)
    sections: list[tuple[str, str]] = []
    intro = parts[0].strip()
    if intro:
        sections.append((title, intro))
    for i in range(1, len(parts), 2):
        heading = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if content:
            sections.append((heading, content))
    return sections


def _chunks() -> list[dict]:
    """Every chunk across the knowledge base, with a stable content hash."""
    chunks: list[dict] = []
    for path in sorted(KB_DIR.glob("*.md")):
        if path.name.startswith("00-"):  # the README is not candidate-facing
            continue
        markdown = path.read_text(encoding="utf-8")
        title = _title(markdown)
        for heading, body in _sections(markdown):
            # Prepend the doc title (and section) so each chunk stands alone.
            label = title if heading == title else f"{title} — {heading}"
            content = f"{label}\n\n{body}"
            digest = hashlib.sha256(
                f"{path.name}|{heading}|{body}".encode("utf-8")
            ).hexdigest()
            chunks.append(
                {
                    "source": path.name,
                    "heading": heading,
                    "content": content,
                    "hash": digest,
                }
            )
    return chunks


def _embed(client, texts: list[str], model: str) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(0, len(texts), _EMBED_BATCH):
        batch = texts[i : i + _EMBED_BATCH]
        out.extend(client.embed(batch, model=model, input_type="document").embeddings)
    return out


def main() -> None:
    settings = get_settings()
    if not settings.voyage_api_key:
        raise SystemExit("Set VOYAGE_API_KEY in the environment.")

    import voyageai

    client = voyageai.Client(api_key=settings.voyage_api_key)
    engine = create_db_engine(settings.database_url)

    desired = {c["hash"]: c for c in _chunks()}

    with engine.begin() as conn:
        existing = {
            row[0] for row in conn.execute(text("select content_hash from kb_chunks"))
        }

        stale = existing - desired.keys()
        if stale:
            conn.execute(
                text("delete from kb_chunks where content_hash = any(:hashes)"),
                {"hashes": list(stale)},
            )

        new = [c for h, c in desired.items() if h not in existing]
        if new:
            embeddings = _embed(
                client, [c["content"] for c in new], settings.embedding_model
            )
            for chunk, embedding in zip(new, embeddings):
                conn.execute(
                    text(
                        "insert into kb_chunks "
                        "(source, heading, content, embedding, content_hash) values "
                        "(:source, :heading, :content, cast(:emb as vector), :hash)"
                    ),
                    {
                        "source": chunk["source"],
                        "heading": chunk["heading"],
                        "content": chunk["content"],
                        "emb": "[" + ",".join(f"{x:.8f}" for x in embedding) + "]",
                        "hash": chunk["hash"],
                    },
                )

    print(
        f"kb_chunks: {len(new)} inserted, {len(stale)} removed, "
        f"{len(desired)} total"
    )


if __name__ == "__main__":
    main()
