from fakes import FakeProvider

from src.config import Settings
from src.fsm.enums import Outcome
from src.fsm.models import ConversationState
from src.orchestrator.engine import ScreeningEngine
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
