from dataclasses import dataclass

from .enums import Action, Outcome, Stage
from .models import CandidateProfile

# Order in which required fields are collected. License and city come early so
# disqualifying candidates are filtered out within the first few turns.
REQUIRED_ORDER: tuple[Stage, ...] = (
    Stage.CONSENT,
    Stage.NAME,
    Stage.LICENSE,
    Stage.CITY,
    Stage.AVAILABILITY,
    Stage.SCHEDULE,
    Stage.EXPERIENCE,
    Stage.START_DATE,
)


def stage_satisfied(stage: Stage, profile: CandidateProfile) -> bool:
    match stage:
        case Stage.CONSENT:
            return profile.consent is True
        case Stage.NAME:
            return bool(profile.full_name)
        case Stage.LICENSE:
            return profile.has_license is True
        case Stage.CITY:
            return bool(profile.city) and profile.city_in_service_area is True
        case Stage.AVAILABILITY:
            return profile.availability is not None
        case Stage.SCHEDULE:
            return profile.preferred_schedule is not None
        case Stage.EXPERIENCE:
            return profile.experience.years is not None
        case Stage.START_DATE:
            return bool(profile.start_date_text)
        case _:
            return True


def next_missing_stage(profile: CandidateProfile) -> Stage | None:
    for stage in REQUIRED_ORDER:
        if not stage_satisfied(stage, profile):
            return stage
    return None


@dataclass(frozen=True)
class Decision:
    action: Action
    stage: Stage | None = None
    outcome: Outcome | None = None
