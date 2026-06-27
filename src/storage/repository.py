import uuid

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from ..fsm.enums import Outcome
from ..fsm.models import ConversationState
from .db import Base
from .models import ConversationRow


class ConversationRepository:
    def __init__(self, database_url: str) -> None:
        self._engine = create_engine(database_url, future=True)
        Base.metadata.create_all(self._engine)
        self._session_factory = sessionmaker(self._engine, future=True)

    def get_or_create(self, candidate_id: str) -> ConversationState:
        with self._session_factory() as session:
            row = session.scalars(
                select(ConversationRow).where(
                    ConversationRow.candidate_id == candidate_id,
                    ConversationRow.outcome == Outcome.IN_PROGRESS.value,
                )
            ).first()
            if row is not None:
                return ConversationState.model_validate(row.state)
        return ConversationState(
            conversation_id=str(uuid.uuid4()), candidate_id=candidate_id
        )

    def save(self, state: ConversationState) -> None:
        payload = state.model_dump(mode="json")
        with self._session_factory() as session:
            row = session.get(ConversationRow, state.conversation_id)
            if row is None:
                row = ConversationRow(conversation_id=state.conversation_id)
                session.add(row)
            row.candidate_id = state.candidate_id
            row.outcome = state.outcome.value
            row.state = payload
            row.updated_at = state.updated_at
            session.commit()

    def list_by_outcome(self, outcome: Outcome) -> list[ConversationState]:
        with self._session_factory() as session:
            rows = session.scalars(
                select(ConversationRow).where(
                    ConversationRow.outcome == outcome.value
                )
            ).all()
            return [ConversationState.model_validate(r.state) for r in rows]
