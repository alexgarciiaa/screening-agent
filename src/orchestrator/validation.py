import re

from ..agents.schemas import TurnUnderstanding
from ..data.service_areas import find_service_area
from ..fsm.enums import Stage
from ..fsm.flow import REQUIRED_ORDER
from ..fsm.models import CandidateProfile

_MULTI_CITY_SPLIT = re.compile(r"\s+(?:y|and|\/|,)\s+", re.IGNORECASE)


def has_multiple_cities(city_text: str) -> bool:
    """True when the candidate appears to name more than one city."""
    parts = [p.strip() for p in _MULTI_CITY_SPLIT.split(city_text.strip()) if p.strip()]
    return len(parts) > 1


def stages_with_extracted_fields(understanding: TurnUnderstanding) -> set[Stage]:
    """Stages that have at least one field present in the latest understanding."""
    stages: set[Stage] = set()
    if understanding.consent is not None:
        stages.add(Stage.CONSENT)
    if understanding.full_name:
        stages.add(Stage.NAME)
    if understanding.has_license is not None:
        stages.add(Stage.LICENSE)
    if understanding.city:
        stages.add(Stage.CITY)
    if understanding.availability is not None:
        stages.add(Stage.AVAILABILITY)
    if understanding.preferred_schedule is not None:
        stages.add(Stage.SCHEDULE)
    if (
        understanding.experience_years is not None
        or understanding.experience_platforms
    ):
        stages.add(Stage.EXPERIENCE)
    if understanding.start_date_text:
        stages.add(Stage.START_DATE)
    return stages


def _stage_order(stage: Stage) -> int:
    try:
        return REQUIRED_ORDER.index(stage)
    except ValueError:
        return len(REQUIRED_ORDER)


def _stages_to_merge(extracted: set[Stage], pending: Stage) -> set[Stage]:
    """Pending stage plus any later stages volunteered in the same message."""
    pending_order = _stage_order(pending)
    return {s for s in extracted if _stage_order(s) >= pending_order}


def apply_understanding(
    profile: CandidateProfile,
    understanding: TurnUnderstanding,
    *,
    pending: Stage | None,
) -> None:
    """Merge fields from the latest message into the profile.

    The pending stage must still be satisfied before the flow moves on, but
    fields from later stages are stored when the candidate volunteers them early.
    When every required stage is satisfied (``pending is None``), any provided
    correction is accepted.
    """
    extracted = stages_with_extracted_fields(understanding)

    if pending is None:
        _merge_stages(profile, understanding, extracted)
        return

    if not extracted:
        return

    _merge_stages(profile, understanding, _stages_to_merge(extracted, pending))


def _merge_stages(
    profile: CandidateProfile,
    understanding: TurnUnderstanding,
    stages: set[Stage],
) -> None:
    for stage in stages:
        _merge_stage(profile, understanding, stage)
    if understanding.vehicle_type:
        profile.vehicle_type = understanding.vehicle_type.strip()


def _merge_stage(
    profile: CandidateProfile,
    understanding: TurnUnderstanding,
    stage: Stage,
) -> None:
    match stage:
        case Stage.CONSENT:
            if understanding.consent is not None and profile.consent is not True:
                profile.consent = understanding.consent
        case Stage.NAME:
            if understanding.full_name:
                profile.full_name = understanding.full_name.strip()
        case Stage.LICENSE:
            if understanding.has_license is not None:
                profile.has_license = understanding.has_license
        case Stage.CITY:
            if understanding.city and not has_multiple_cities(understanding.city):
                profile.city = understanding.city.strip()
                profile.matched_location = None
                profile.city_in_service_area = None
        case Stage.AVAILABILITY:
            if understanding.availability is not None:
                profile.availability = understanding.availability
        case Stage.SCHEDULE:
            if understanding.preferred_schedule is not None:
                profile.preferred_schedule = understanding.preferred_schedule
        case Stage.EXPERIENCE:
            if understanding.experience_years is not None:
                profile.experience.years = understanding.experience_years
            if understanding.experience_platforms:
                profile.experience.platforms = understanding.experience_platforms
        case Stage.START_DATE:
            if understanding.start_date_text:
                profile.start_date_text = understanding.start_date_text.strip()


def resolve_service_area(profile: CandidateProfile) -> None:
    """Match a freshly provided city against the serviced locations."""
    if not profile.city or profile.city_in_service_area is not None:
        return
    match = find_service_area(profile.city)
    profile.matched_location = match.city if match else None
    profile.city_in_service_area = match is not None
