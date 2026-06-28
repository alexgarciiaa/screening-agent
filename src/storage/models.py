from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


class ConversationRow(Base):
    __tablename__ = "conversations"

    conversation_id: Mapped[str] = mapped_column(String, primary_key=True)
    candidate_id: Mapped[str] = mapped_column(String, index=True)
    outcome: Mapped[str] = mapped_column(String, index=True)
    # Flat columns so the conversation is readable in a dashboard; the full
    # state lives in `state` for resuming.
    full_name: Mapped[str | None] = mapped_column(String, nullable=True)
    city: Mapped[str | None] = mapped_column(String, nullable=True)
    qualified: Mapped[bool] = mapped_column(Boolean, default=False)
    state: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime)


class ServiceAreaRow(Base):
    """A city Grupo Sazon operates in. Editable by HR from the dashboard."""

    __tablename__ = "service_areas"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    city: Mapped[str] = mapped_column(String)
    country: Mapped[str] = mapped_column(String)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
