"""Service-area lookup for the cities Grupo Sazon operates in.

This is a representative subset of the 45 real locations across Spain and
Mexico. In production this table would be loaded from the client's systems.
"""

from dataclasses import dataclass

from rapidfuzz import fuzz, process

# Score below which a candidate's city is treated as not matching any
# serviced location. Tuned to tolerate typos and accents without matching
# unrelated city names.
_MATCH_THRESHOLD = 82.0


@dataclass(frozen=True)
class ServiceArea:
    city: str
    country: str


SERVICE_AREAS: tuple[ServiceArea, ...] = (
    ServiceArea("Madrid", "ES"),
    ServiceArea("Barcelona", "ES"),
    ServiceArea("Valencia", "ES"),
    ServiceArea("Sevilla", "ES"),
    ServiceArea("Zaragoza", "ES"),
    ServiceArea("Malaga", "ES"),
    ServiceArea("Bilbao", "ES"),
    ServiceArea("Ciudad de Mexico", "MX"),
    ServiceArea("Guadalajara", "MX"),
    ServiceArea("Monterrey", "MX"),
    ServiceArea("Puebla", "MX"),
    ServiceArea("Queretaro", "MX"),
    ServiceArea("Tijuana", "MX"),
    ServiceArea("Cancun", "MX"),
)

_ALIASES = {
    "cdmx": "Ciudad de Mexico",
    "mexico df": "Ciudad de Mexico",
    "df": "Ciudad de Mexico",
}

_BY_NAME = {area.city.lower(): area for area in SERVICE_AREAS}
_CHOICES = list(_BY_NAME.keys())


def find_service_area(city_text: str) -> ServiceArea | None:
    """Return the serviced location best matching the candidate's input, or None."""
    query = city_text.strip().lower()
    if not query:
        return None

    query = _ALIASES.get(query, query)

    match = process.extractOne(query, _CHOICES, scorer=fuzz.WRatio)
    if match is None or match[1] < _MATCH_THRESHOLD:
        return None
    return _BY_NAME[match[0]]
