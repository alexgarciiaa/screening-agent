from ..fsm.enums import Action, Intent, Outcome, Stage
from ..fsm.flow import Decision, next_missing_stage
from ..fsm.models import ConversationState


def decide(state: ConversationState) -> Decision:
    """Pick the next action. Gates and field order live here, not in the model."""
    profile = state.profile

    if state.last_intent is Intent.STOP:
        return Decision(Action.CLOSE_OPTED_OUT, outcome=Outcome.OPTED_OUT)

    if profile.consent is False:
        return Decision(
            Action.CLOSE_CONSENT_DECLINED, outcome=Outcome.CONSENT_DECLINED
        )
    if profile.consent is not True:
        return Decision(Action.ASK, stage=Stage.CONSENT)

    if profile.has_license is False:
        return Decision(
            Action.CLOSE_DISQUALIFIED_NO_LICENSE,
            outcome=Outcome.DISQUALIFIED_NO_LICENSE,
        )
    if profile.city and profile.city_in_service_area is False:
        return Decision(Action.CLOSE_OUT_OF_AREA, outcome=Outcome.OUT_OF_AREA)

    stage = next_missing_stage(profile)

    if state.last_intent is Intent.UNCLEAR and stage is not None:
        return Decision(Action.CLARIFY, stage=stage)

    if state.last_intent is Intent.QUESTION:
        return Decision(Action.ANSWER_QUESTION, stage=stage)

    if stage is not None:
        return Decision(Action.ASK, stage=stage)

    if state.awaiting_confirmation:
        if state.last_confirmation is True:
            return Decision(Action.CLOSE_QUALIFIED, outcome=Outcome.QUALIFIED)
        if state.last_confirmation is False:
            return Decision(Action.CLARIFY, stage=Stage.SUMMARY)
    return Decision(Action.CONFIRM_SUMMARY, stage=Stage.SUMMARY)
