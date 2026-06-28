from fakes import FakeProvider

from src.config import Settings
from src.fsm.enums import Outcome
from src.fsm.models import ConversationState
from src.orchestrator.engine import ScreeningEngine
from src.orchestrator.handoff import build_handoff
from src.storage.repository import ConversationRepository


def make_engine() -> ScreeningEngine:
    settings = Settings(anthropic_api_key=None, database_url="sqlite://")
    return ScreeningEngine(FakeProvider(settings), settings)


def new_state() -> ConversationState:
    return ConversationState(conversation_id="t", candidate_id="c")


def run(engine: ScreeningEngine, state: ConversationState, messages: list[str]) -> None:
    engine.start(state)
    for message in messages:
        engine.handle(state, message)


def test_happy_path_qualifies():
    engine, state = make_engine(), new_state()
    run(
        engine,
        state,
        [
            "si",
            "Laura Gomez",
            "si",
            "Madrid",
            "jornada completa",
            "por la manana",
            "2 anos en Glovo y Uber Eats",
            "el lunes que viene",
            "si",
        ],
    )
    assert state.outcome is Outcome.QUALIFIED
    assert state.profile.full_name == "Laura Gomez"
    assert state.profile.city_in_service_area is True
    assert state.profile.experience.years == 2.0


def test_no_license_disqualifies():
    engine, state = make_engine(), new_state()
    run(engine, state, ["si", "Pedro", "no"])
    assert state.outcome is Outcome.DISQUALIFIED_NO_LICENSE


def test_out_of_area_routes_to_waitlist():
    engine, state = make_engine(), new_state()
    run(engine, state, ["si", "Ana", "si", "Toledo"])
    assert state.outcome is Outcome.OUT_OF_AREA
    assert state.profile.city_in_service_area is False


def test_stop_opts_out():
    engine, state = make_engine(), new_state()
    run(engine, state, ["para por favor"])
    assert state.outcome is Outcome.OPTED_OUT


def test_repository_round_trip(tmp_path):
    repo = ConversationRepository(f"sqlite:///{tmp_path / 'test.db'}")
    state = repo.get_or_create("c1")
    state.add_message("agent", "hola")
    repo.save(state)

    resumed = repo.get_or_create("c1")
    assert resumed.conversation_id == state.conversation_id
    assert resumed.messages[-1].text == "hola"


def test_reset_starts_a_fresh_conversation(tmp_path):
    repo = ConversationRepository(f"sqlite:///{tmp_path / 'r.db'}")
    state = repo.get_or_create("c1")
    state.add_message("agent", "hola")
    repo.save(state)

    repo.reset("c1")
    fresh = repo.get_or_create("c1")
    assert fresh.conversation_id != state.conversation_id
    assert fresh.messages == []


def test_dashboard_columns_are_populated(tmp_path):
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    from src.storage.models import ConversationRow

    repo = ConversationRepository(f"sqlite:///{tmp_path / 'd.db'}")
    state = repo.get_or_create("c1")
    state.profile.full_name = "Laura"
    state.profile.city = "Madrid"
    state.outcome = Outcome.QUALIFIED
    repo.save(state)

    with Session(repo._engine) as session:
        row = session.scalars(select(ConversationRow)).one()
    assert row.full_name == "Laura"
    assert row.city == "Madrid"
    assert row.qualified is True


def test_summary_rejection_keeps_consent_and_stays_open():
    engine, state = make_engine(), new_state()
    run(
        engine,
        state,
        [
            "si",
            "Laura Gomez",
            "si",
            "Madrid",
            "jornada completa",
            "por la manana",
            "2 anos en Glovo",
            "el lunes",
        ],
    )
    engine.handle(state, "no")
    assert state.outcome is Outcome.IN_PROGRESS
    assert state.profile.consent is True


def test_terminal_outcome_is_absorbing():
    engine, state = make_engine(), new_state()
    run(engine, state, ["si", "Pedro", "no"])
    assert state.outcome is Outcome.DISQUALIFIED_NO_LICENSE

    before = len(state.messages)
    turn = engine.handle(state, "y esto que?")
    assert turn.finished is True
    assert state.outcome is Outcome.DISQUALIFIED_NO_LICENSE
    assert len(state.messages) == before


def test_handoff_qualified():
    engine, state = make_engine(), new_state()
    run(
        engine,
        state,
        [
            "si",
            "Laura Gomez",
            "si",
            "Madrid",
            "jornada completa",
            "por la manana",
            "2 anos en Glovo",
            "el lunes",
            "si",
        ],
    )
    handoff = build_handoff(state)
    assert handoff.qualified is True
    assert handoff.outcome is Outcome.QUALIFIED
    assert "Laura Gomez" in handoff.summary
    assert "entrevista" in handoff.recommended_action.lower()


def test_handoff_disqualified():
    engine, state = make_engine(), new_state()
    run(engine, state, ["si", "Pedro", "no"])
    handoff = build_handoff(state)
    assert handoff.qualified is False
    assert handoff.outcome is Outcome.DISQUALIFIED_NO_LICENSE
    assert "carnet" in handoff.recommended_action.lower()
