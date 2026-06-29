from ..config import get_settings
from ..fsm.enums import Action, Language, Stage
from ..fsm.flow import Decision, next_missing_stage
from ..fsm.models import CandidateProfile, ConversationState
from ..orchestrator.validation import has_multiple_cities

SYSTEM_UNDERSTAND = """\
You read one message from a candidate applying to be a delivery driver and \
return structured data.

Extract only the information explicitly present in the candidate's latest \
message. Leave every other field null; do not guess or carry over earlier \
answers. Detect the language of that message (es or en), classify the intent \
(answer, question, chitchat, stop, unclear) and the sentiment (neutral, \
positive, frustrated, confused).

Guidance:
- has_license is false only if the candidate clearly states they have no \
driving licence.
- city is the city or zone exactly as the candidate writes it.
- experience_years is a number; experience_platforms lists named apps \
(Glovo, Uber Eats, Rappi, ...).
- do not detect the language of the message based on the name of a business or city; use the language of the text.
- consent is true/false only when the candidate answers the consent question.
- confirmation is true if the candidate agrees the summary is correct, false if \
they want to change something; otherwise null.
- If the message is gibberish or impossible to interpret, set intent to unclear.
"""

SYSTEM_REPLY = """\
You are the hiring assistant for Grupo Sazon, a restaurant chain, screening \
candidates for delivery-driver roles over messaging.

Every message follows the same pattern:
1. A brief, warm acknowledgement of the candidate's last answer (a few words, \
e.g. "Genial", "Perfecto, gracias"). Skip this only on the very first message.
2. Then do exactly what the task says, normally a single question.
Keep both parts to one or two short sentences that fit on a phone screen.

Rules:
- Write in the candidate's language (Spanish or English) and match their \
register, including Spain vs Mexico variants.
- One question per message; never stack two.
- Try to include one emoji, only when it fits naturally.
- Try to use the candidate's name in the message, if known.
- Never invent salary, schedules or commitments; defer specifics to the \
recruiter.
- After the first message never greet again, never restart the screening, and \
never re-ask something already answered.

Output only the message to send. No labels, no quotes, no notes.
"""

_ASK_DIRECTIVES: dict[Stage, str] = {
    Stage.CONSENT: (
        "Greet the candidate, say this is a quick screening for the "
        "delivery-driver role (about 2 minutes), and ask if they are happy to "
        "continue."
    ),
    Stage.NAME: "Ask for the candidate's full name.",
    Stage.LICENSE: "Ask whether the candidate holds a valid driving licence.",
    Stage.CITY: "Ask in which city or zone the candidate wants to work.",
    Stage.AVAILABILITY: (
        "Ask about availability: full-time, part-time or weekends."
    ),
    Stage.SCHEDULE: (
        "Ask about preferred schedule: morning, afternoon, evening or flexible."
    ),
    Stage.EXPERIENCE: (
        "Ask about prior delivery experience: how many years and on which "
        "platforms (Glovo, Uber Eats, Rappi...)."
    ),
    Stage.START_DATE: "Ask when the candidate could start.",
}

_CLOSE_DIRECTIVES: dict[Action, str] = {
    Action.CLOSE_QUALIFIED: (
        "Thank the candidate warmly, confirm their profile looks like a good "
        "fit, and tell them a recruiter will contact them shortly to continue"
    ),
    Action.CLOSE_DISQUALIFIED_NO_LICENSE: (
        "Kindly explain that a valid driving licence is required for this role, "
        "so you cannot continue for now, and that they are welcome to re-apply "
        "if that changes."
    ),
    Action.CLOSE_OUT_OF_AREA: (
        "Explain that Grupo Sazon does not operate in their area yet, offer to "
        "keep their details for when it expands there, and thank them."
    ),
    Action.CLOSE_CONSENT_DECLINED: (
        "Respectfully acknowledge that they prefer not to continue and wish "
        "them well."
    ),
    Action.CLOSE_OPTED_OUT: (
        "Confirm that you will stop here and that they can reply anytime to "
        "pick up again."
    ),
}


def format_transcript(state: ConversationState, limit: int | None = None) -> str:
    limit = limit or get_settings().history_turns_in_context
    recent = state.messages[-limit:]
    lines = [f"{m.role.capitalize()}: {m.text}" for m in recent]
    return "\n".join(lines)


def profile_summary(profile: CandidateProfile) -> str:
    known: list[str] = []
    if profile.full_name:
        known.append(f"name={profile.full_name}")
    if profile.has_license is not None:
        known.append(f"licence={'yes' if profile.has_license else 'no'}")
    if profile.city:
        known.append(f"city={profile.city}")
    if profile.availability:
        known.append(f"availability={profile.availability.value}")
    if profile.preferred_schedule:
        known.append(f"schedule={profile.preferred_schedule.value}")
    if profile.experience.years is not None:
        platforms = ", ".join(profile.experience.platforms) or "none"
        known.append(f"experience={profile.experience.years}y ({platforms})")
    if profile.start_date_text:
        known.append(f"start={profile.start_date_text}")
    return "; ".join(known) if known else "nothing yet"


def understand_user_message(state: ConversationState) -> str:
    pending = next_missing_stage(state.profile)
    if state.last_asked_stage is not None:
        asked_about = state.last_asked_stage.value
    elif pending is not None:
        asked_about = pending.value
    else:
        asked_about = "confirming the summary"
    return (
        "Conversation so far:\n"
        f"{format_transcript(state)}\n\n"
        f"The assistant's last question was about: {asked_about}.\n"
        "Classify the candidate's latest message and extract any fields it "
        "provides. If it answers the question above, fill that field."
    )


def _directive(state: ConversationState, decision: Decision) -> str:
    if decision.action is Action.ASK and decision.stage is not None:
        return _ASK_DIRECTIVES[decision.stage]
    if decision.action is Action.CLARIFY and decision.stage is Stage.SUMMARY:
        return (
            "The candidate says the summary is not correct. Ask warmly which "
            "detail they would like to change."
        )
    if decision.action is Action.CLARIFY and decision.stage is not None:
        if (
            decision.stage is Stage.CITY
            and has_multiple_cities(
                next(
                    (m.text for m in reversed(state.messages) if m.role == "candidate"),
                    "",
                )
            )
        ):
            return (
                "The candidate gave more than one city. Ask them to choose a single "
                "city or zone where they want to work."
            )
        return (
            "The previous answer was unclear. Re-ask, more concretely and with "
            f"example options. {_ASK_DIRECTIVES[decision.stage]}"
        )
    if decision.action is Action.ANSWER_QUESTION:
        pending = next_missing_stage(state.profile)
        follow_up = _ASK_DIRECTIVES.get(pending, "") if pending else ""
        return (
            "Answer the candidate's question briefly and honestly, using the "
            "company information provided below if available and deferring exact "
            f"figures to the recruiter. Then continue: {follow_up}"
        )
    if decision.action is Action.CONFIRM_SUMMARY:
        return (
            "Summarise the collected information in one short message and ask "
            "the candidate to confirm it is correct."
        )
    return _CLOSE_DIRECTIVES.get(decision.action, "Reply politely.")


def reply_user_message(
    state: ConversationState, decision: Decision, context: str | None = None
) -> str:
    language = "Spanish" if state.language is Language.ES else "English"
    knowledge = (
        "Company information you can use to answer "
        f"(do not invent anything beyond it):\n{context}\n\n"
        if context
        else ""
    )
    return (
        f"Task (do exactly this, nothing else): {_directive(state, decision)}\n\n"
        f"Already known, do not ask again: {profile_summary(state.profile)}\n\n"
        f"{knowledge}"
        "Recent messages:\n"
        f"{format_transcript(state, limit=4)}\n\n"
        f"Write one short message in {language}."
    )
