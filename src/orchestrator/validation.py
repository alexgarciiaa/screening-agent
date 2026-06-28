from ..agents.schemas import TurnUnderstanding
from ..data.service_areas import find_service_area
from ..fsm.enums import Stage
from ..fsm.models import CandidateProfile


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


def apply_understanding(
    profile: CandidateProfile,
    understanding: TurnUnderstanding,
    *,
    pending: Stage | None,
) -> None:
    """Merge fields from the latest message into the profile.

    By default only the pending collection stage is accepted. Multiple stages in
    one message are merged together (voluntary multi-answer). When every required
    stage is satisfied (``pending is None``), any provided correction is accepted.
    """
    extracted = stages_with_extracted_fields(understanding)

    if pending is None:
        _merge_stages(profile, understanding, extracted)
        return

    if len(extracted) > 1:
        _merge_stages(profile, understanding, extracted)
        return

    if extracted and extracted != {pending}:
        return

    _merge_stages(profile, understanding, {pending})


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
            if understanding.city:
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
