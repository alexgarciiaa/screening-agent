"""Console runner for the screening agent. Exercises the full flow end to end."""

from .config import get_settings
from .fsm.enums import TERMINAL_OUTCOMES
from .agents.provider import build_provider
from .agents.retrieval import build_retriever
from .data import service_areas
from .orchestrator.engine import ScreeningEngine
from .orchestrator.handoff import build_handoff
from .storage.repository import ConversationRepository


def main() -> None:
    settings = get_settings()
    try:
        provider = build_provider(settings)
    except RuntimeError as exc:
        print(exc)
        return
    service_areas.configure(settings.database_url)
    repository = ConversationRepository(settings.database_url)
    engine = ScreeningEngine(provider, build_retriever(settings))

    candidate_id = input("Candidate phone/id [demo]: ").strip() or "demo"
    state = repository.get_or_create(candidate_id)

    if state.messages:
        print("\nResuming the existing conversation.\n")
        print(f"Agent: {state.messages[-1].text}\n")
    else:
        greeting = engine.start(state)
        repository.save(state)
        print(f"\nAgent: {greeting}\n")

    while state.outcome not in TERMINAL_OUTCOMES:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not text:
            continue

        try:
            turn = engine.handle(state, text)
        except Exception as exc:  # surface the failure without crashing the loop
            print(f"\n[error handling message: {exc}]\n")
            continue

        repository.save(state)
        print(f"\nAgent: {turn.reply}\n")

    print(f"[conversation closed: {state.outcome.value}]\n")
    print("Recruiter handoff:")
    print(build_handoff(state).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
