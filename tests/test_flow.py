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


def test_finished_conversation_is_not_active(tmp_path):
    repo = ConversationRepository(f"sqlite:///{tmp_path / 'a.db'}")
    state = repo.get_or_create("c1")
    state.add_message("agent", "hola")
    repo.save(state)
    assert repo.get_active("c1") is not None

    state.outcome = Outcome.QUALIFIED
    repo.save(state)
    assert repo.get_active("c1") is None
    latest = repo.latest("c1")
    assert latest is not None and latest.outcome is Outcome.QUALIFIED


def test_service_area_catalog_seeds_and_matches(tmp_path):
    from src.data.service_areas import ServiceAreaCatalog

    catalog = ServiceAreaCatalog(f"sqlite:///{tmp_path / 'sa.db'}")
    assert catalog.find("Madrid") is not None
    assert catalog.find("cdmx").city == "Ciudad de Mexico"  # alias
    assert catalog.find("Toledo") is None


def test_voice_modality_is_recorded():
    from src.fsm.enums import Modality

    engine, state = make_engine(), new_state()
    engine.start(state)
    engine.handle(state, "si", Modality.VOICE)
    candidate_msg = state.messages[1]
    assert candidate_msg.role == "candidate"
    assert candidate_msg.modality is Modality.VOICE


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


def test_start_sets_last_asked_stage():
    engine, state = make_engine(), new_state()
    engine.start(state)
    from src.fsm.enums import Stage

    assert state.last_asked_stage is Stage.CONSENT


def test_wrong_stage_field_is_ignored():
    from src.agents.schemas import TurnUnderstanding
    from src.fsm.enums import Intent, Language, Sentiment, Stage
    from src.fsm.models import CandidateProfile
    from src.orchestrator.validation import apply_understanding

    profile = CandidateProfile(consent=True)
    understanding = TurnUnderstanding(
        language=Language.ES,
        intent=Intent.ANSWER,
        sentiment=Sentiment.NEUTRAL,
        city="Madrid",
    )
    apply_understanding(profile, understanding, pending=Stage.NAME)
    assert profile.city is None


def test_multi_field_message_is_accepted():
    from src.agents.schemas import TurnUnderstanding
    from src.fsm.enums import Intent, Language, Sentiment, Stage
    from src.fsm.models import CandidateProfile
    from src.orchestrator.validation import apply_understanding

    profile = CandidateProfile()
    understanding = TurnUnderstanding(
        language=Language.ES,
        intent=Intent.ANSWER,
        sentiment=Sentiment.NEUTRAL,
        consent=True,
        full_name="Laura Gomez",
    )
    apply_understanding(profile, understanding, pending=Stage.CONSENT)
    assert profile.consent is True
    assert profile.full_name == "Laura Gomez"


def test_summary_correction_updates_profile():
    from src.agents.schemas import TurnUnderstanding
    from src.fsm.enums import Intent, Language, Sentiment
    from src.fsm.models import CandidateProfile
    from src.orchestrator.validation import apply_understanding

    profile = CandidateProfile(
        consent=True,
        full_name="Laura",
        has_license=True,
        city="Madrid",
        city_in_service_area=True,
    )
    understanding = TurnUnderstanding(
        language=Language.ES,
        intent=Intent.ANSWER,
        sentiment=Sentiment.NEUTRAL,
        city="Barcelona",
    )
    apply_understanding(profile, understanding, pending=None)
    assert profile.city == "Barcelona"
    assert profile.city_in_service_area is None


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
