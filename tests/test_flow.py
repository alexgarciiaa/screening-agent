from fakes import FakeProvider

from src.config import Settings
from src.fsm.enums import Outcome
from src.fsm.models import ConversationState
from src.orchestrator.engine import ScreeningEngine
from src.orchestrator.handoff import build_handoff
from src.storage.repository import ConversationRepository


def make_engine() -> ScreeningEngine:
    settings = Settings(anthropic_api_key=None, database_url="sqlite://")
    return ScreeningEngine(FakeProvider(settings))


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


def test_lookahead_saves_name_during_consent():
    from src.agents.schemas import TurnUnderstanding
    from src.fsm.enums import Intent, Language, Sentiment, Stage
    from src.fsm.models import CandidateProfile
    from src.orchestrator.validation import apply_understanding

    profile = CandidateProfile()
    understanding = TurnUnderstanding(
        language=Language.ES,
        intent=Intent.ANSWER,
        sentiment=Sentiment.NEUTRAL,
        full_name="Juan Lopez Garrido",
    )
    apply_understanding(profile, understanding, pending=Stage.CONSENT)
    assert profile.consent is None
    assert profile.full_name == "Juan Lopez Garrido"


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


def test_multiple_cities_are_not_saved():
    from src.agents.schemas import TurnUnderstanding
    from src.fsm.enums import Intent, Language, Sentiment, Stage
    from src.fsm.models import CandidateProfile
    from src.orchestrator.validation import apply_understanding, has_multiple_cities

    assert has_multiple_cities("madrid y buenos aires")

    profile = CandidateProfile(consent=True, full_name="Juan", has_license=True)
    understanding = TurnUnderstanding(
        language=Language.ES,
        intent=Intent.ANSWER,
        sentiment=Sentiment.NEUTRAL,
        city="madrid y buenos aires",
    )
    apply_understanding(profile, understanding, pending=Stage.CITY)
    assert profile.city is None


def test_multiple_cities_trigger_clarify():
    from src.fsm.enums import Action, Stage

    engine, state = make_engine(), new_state()
    run(engine, state, ["si", "Juan Lopez", "si"])
    turn = engine.handle(state, "madrid y buenos aires")
    assert state.profile.city is None
    assert turn.decision is not None
    assert turn.decision.action is Action.CLARIFY
    assert turn.decision.stage is Stage.CITY


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


def test_nps_followup_after_finish():
    engine, state = make_engine(), new_state()
    run(engine, state, ["si", "Pedro", "no"])
    assert state.outcome is Outcome.DISQUALIFIED_NO_LICENSE

    first = engine.follow_up(state, "muchas gracias")
    assert first is not None
    assert state.nps_asked is True
    assert state.nps_score is None

    engine.follow_up(state, "le doy un 8")
    assert state.nps_score == 8
    assert state.nps_done is True

    assert engine.follow_up(state, "vale, adios") is None


def test_nps_out_of_range_is_reasked():
    engine, state = make_engine(), new_state()
    run(engine, state, ["si", "Pedro", "no"])

    engine.follow_up(state, "gracias")
    again = engine.follow_up(state, "un 12")  # out of range
    assert again is not None
    assert state.nps_score is None
    assert state.nps_done is False

    engine.follow_up(state, "pues un 9")
    assert state.nps_score == 9
    assert state.nps_done is True


def test_nps_classification():
    from src.fsm.enums import NpsCategory, classify_nps

    assert classify_nps(10) is NpsCategory.PROMOTER
    assert classify_nps(9) is NpsCategory.PROMOTER
    assert classify_nps(8) is NpsCategory.PASSIVE
    assert classify_nps(7) is NpsCategory.PASSIVE
    assert classify_nps(6) is NpsCategory.DETRACTOR
    assert classify_nps(0) is NpsCategory.DETRACTOR


def test_nps_category_stored_for_dashboard(tmp_path):
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    from src.storage.models import ConversationRow

    repo = ConversationRepository(f"sqlite:///{tmp_path / 'nps.db'}")
    state = repo.get_or_create("c1")
    state.nps_score = 9
    repo.save(state)

    with Session(repo._engine) as session:
        row = session.scalars(select(ConversationRow)).one()
    assert row.nps == 9
    assert row.nps_category == "promoter"


def test_reminders_fire_after_2h_and_24h():
    from datetime import timedelta

    engine, state = make_engine(), new_state()
    engine.start(state)
    engine.handle(state, "hola")  # candidate active; starts the silence clock
    base = state.last_candidate_at
    assert base is not None

    assert engine.reminder(state, now=base + timedelta(minutes=30)) is None
    assert engine.reminder(state, now=base + timedelta(hours=2, minutes=1)) is not None
    assert state.reminders_sent == 1
    # already sent; not repeated before the next threshold
    assert engine.reminder(state, now=base + timedelta(hours=5)) is None
    assert engine.reminder(state, now=base + timedelta(hours=24, minutes=1)) is not None
    assert state.reminders_sent == 2
    # only two nudges, ever
    assert engine.reminder(state, now=base + timedelta(days=5)) is None


def test_reminders_reset_when_candidate_replies():
    from datetime import timedelta

    engine, state = make_engine(), new_state()
    engine.start(state)
    engine.handle(state, "hola")
    base = state.last_candidate_at
    engine.reminder(state, now=base + timedelta(hours=2, minutes=1))
    assert state.reminders_sent == 1

    engine.handle(state, "si")  # candidate comes back -> clock resets
    assert state.reminders_sent == 0


def test_no_reminders_after_screening_finished():
    from datetime import timedelta

    engine, state = make_engine(), new_state()
    run(engine, state, ["si", "Pedro", "no"])  # disqualified -> terminal
    base = state.last_candidate_at or state.created_at
    assert engine.reminder(state, now=base + timedelta(days=2)) is None


def test_reply_prompt_includes_retrieved_context():
    from src.agents.prompts import reply_user_message
    from src.fsm.enums import Action
    from src.fsm.flow import Decision

    state, decision = new_state(), Decision(Action.ANSWER_QUESTION)
    with_ctx = reply_user_message(state, decision, context="[06.md] Pago quincenal.")
    assert "Pago quincenal." in with_ctx
    assert "Company information" in with_ctx
    # without context there is no knowledge block
    assert "Company information" not in reply_user_message(state, decision)


def test_retrieve_context_only_for_questions():
    from src.agents.retrieval import RetrievedChunk
    from src.fsm.enums import Action, Stage
    from src.fsm.flow import Decision

    class StubRetriever:
        def search(self, query):
            return [RetrievedChunk("06.md", "Pagos", "Pago quincenal.", 0.9)]

    engine = ScreeningEngine(FakeProvider(), StubRetriever())

    ctx = engine._retrieve_context(Decision(Action.ANSWER_QUESTION), "¿cuándo cobro?")
    assert ctx is not None and "Pago quincenal." in ctx
    # a normal ASK turn does not retrieve
    assert engine._retrieve_context(Decision(Action.ASK, stage=Stage.NAME), "Pedro") is None


def test_no_retriever_means_no_context():
    from src.fsm.enums import Action
    from src.fsm.flow import Decision

    engine = make_engine()  # retriever defaults to None
    assert engine._retrieve_context(Decision(Action.ANSWER_QUESTION), "hola") is None


def test_question_does_not_capture_fields():
    # A field mentioned inside a QUESTION must not be saved as the candidate's.
    from src.agents.schemas import TurnUnderstanding
    from src.fsm.enums import Intent, Language, Sentiment

    class QuestionProvider(FakeProvider):
        def understand(self, state):
            return TurnUnderstanding(
                language=Language.ES,
                intent=Intent.QUESTION,
                sentiment=Sentiment.NEUTRAL,
                city="Barcelona",
            )

    state = new_state()
    state.profile.consent = True  # past the consent gate
    engine = ScreeningEngine(QuestionProvider())

    engine.handle(state, "¿operáis en Barcelona?")

    assert state.profile.city is None


def test_question_before_consent_is_answered():
    from src.fsm.enums import Action, Intent, Stage
    from src.orchestrator.decision import decide

    state = new_state()  # consent not given yet
    state.last_intent = Intent.QUESTION
    d = decide(state)
    assert d.action is Action.ANSWER_QUESTION
    assert d.stage is Stage.CONSENT  # answers the FAQ, then resumes by asking consent

    state.last_intent = Intent.ANSWER  # a non-question still asks for consent
    d2 = decide(state)
    assert d2.action is Action.ASK and d2.stage is Stage.CONSENT
