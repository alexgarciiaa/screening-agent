from ..data.service_areas import find_service_area
from ..fsm.models import CandidateProfile
from ..agents.schemas import TurnUnderstanding


def apply_understanding(
    profile: CandidateProfile, understanding: TurnUnderstanding
) -> None:
    """Merge the fields present in the latest message, keeping known values."""
    if understanding.consent is not None and profile.consent is not True:
        profile.consent = understanding.consent
    if understanding.full_name:
        profile.full_name = understanding.full_name.strip()
    if understanding.has_license is not None:
        profile.has_license = understanding.has_license
    if understanding.city:
        profile.city = understanding.city.strip()
        profile.matched_location = None
        profile.city_in_service_area = None
    if understanding.availability is not None:
        profile.availability = understanding.availability
    if understanding.preferred_schedule is not None:
        profile.preferred_schedule = understanding.preferred_schedule
    if understanding.experience_years is not None:
        profile.experience.years = understanding.experience_years
    if understanding.experience_platforms:
        profile.experience.platforms = understanding.experience_platforms
    if understanding.start_date_text:
        profile.start_date_text = understanding.start_date_text.strip()
    if understanding.vehicle_type:
        profile.vehicle_type = understanding.vehicle_type.strip()


def resolve_service_area(profile: CandidateProfile) -> None:
    """Match a freshly provided city against the serviced locations."""
    if not profile.city or profile.city_in_service_area is not None:
        return
    match = find_service_area(profile.city)
    profile.matched_location = match.city if match else None
    profile.city_in_service_area = match is not None
