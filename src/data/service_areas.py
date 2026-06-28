"""Service-area lookup for the cities Grupo Sazon operates in.

Cities live in the database (table `service_areas`) so HR can add or remove them
without touching the code. The list below is the seed loaded on first run and
the fallback used when no database is configured. City names are kept
accent-free so candidate typos and unaccented input still match.
"""

import time
from dataclasses import dataclass

from rapidfuzz import fuzz, process
from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

from ..storage.db import Base, create_db_engine
from ..storage.models import ServiceAreaRow

# Score below which a candidate's city is treated as not matching any serviced
# location. Tuned to tolerate typos without matching unrelated city names.
_MATCH_THRESHOLD = 82.0
_CACHE_TTL_SECONDS = 300


@dataclass(frozen=True)
class ServiceArea:
    city: str
    country: str


_SEED: tuple[ServiceArea, ...] = (
    # Spain
    ServiceArea("Madrid", "ES"),
    ServiceArea("Barcelona", "ES"),
    ServiceArea("Valencia", "ES"),
    ServiceArea("Sevilla", "ES"),
    ServiceArea("Zaragoza", "ES"),
    ServiceArea("Malaga", "ES"),
    ServiceArea("Murcia", "ES"),
    ServiceArea("Palma", "ES"),
    ServiceArea("Las Palmas", "ES"),
    ServiceArea("Bilbao", "ES"),
    ServiceArea("Alicante", "ES"),
    ServiceArea("Cordoba", "ES"),
    ServiceArea("Valladolid", "ES"),
    ServiceArea("Vigo", "ES"),
    ServiceArea("Gijon", "ES"),
    ServiceArea("Granada", "ES"),
    ServiceArea("A Coruna", "ES"),
    ServiceArea("Vitoria", "ES"),
    ServiceArea("Pamplona", "ES"),
    ServiceArea("Almeria", "ES"),
    ServiceArea("San Sebastian", "ES"),
    ServiceArea("Santander", "ES"),
    # Mexico
    ServiceArea("Ciudad de Mexico", "MX"),
    ServiceArea("Guadalajara", "MX"),
    ServiceArea("Monterrey", "MX"),
    ServiceArea("Puebla", "MX"),
    ServiceArea("Tijuana", "MX"),
    ServiceArea("Leon", "MX"),
    ServiceArea("Queretaro", "MX"),
    ServiceArea("Ciudad Juarez", "MX"),
    ServiceArea("Zapopan", "MX"),
    ServiceArea("Merida", "MX"),
    ServiceArea("Cancun", "MX"),
    ServiceArea("San Luis Potosi", "MX"),
    ServiceArea("Aguascalientes", "MX"),
    ServiceArea("Saltillo", "MX"),
    ServiceArea("Mexicali", "MX"),
    ServiceArea("Culiacan", "MX"),
    ServiceArea("Hermosillo", "MX"),
    ServiceArea("Toluca", "MX"),
    ServiceArea("Chihuahua", "MX"),
    ServiceArea("Morelia", "MX"),
    ServiceArea("Veracruz", "MX"),
    ServiceArea("Cuernavaca", "MX"),
    ServiceArea("Acapulco", "MX"),
)

_ALIASES = {
    "cdmx": "ciudad de mexico",
    "mexico df": "ciudad de mexico",
    "df": "ciudad de mexico",
}

_SEED_BY_NAME = {area.city.lower(): area for area in _SEED}


def _match(city_text: str, by_name: dict[str, ServiceArea]) -> ServiceArea | None:
    query = city_text.strip().lower()
    if not query:
        return None
    query = _ALIASES.get(query, query)
    choices = list(by_name)
    if not choices:
        return None
    match = process.extractOne(query, choices, scorer=fuzz.WRatio)
    if match is None or match[1] < _MATCH_THRESHOLD:
        return None
    return by_name[match[0]]


class ServiceAreaCatalog:
    """Database-backed service areas, seeded from `_SEED` and cached briefly."""

    def __init__(self, database_url: str) -> None:
        self._engine = create_db_engine(database_url)
        Base.metadata.create_all(self._engine)
        self._session_factory = sessionmaker(self._engine, future=True)
        self._seed_if_empty()
        self._by_name: dict[str, ServiceArea] = {}
        self._loaded_at = 0.0

    def _seed_if_empty(self) -> None:
        with self._session_factory() as session:
            if session.scalar(select(func.count()).select_from(ServiceAreaRow)) == 0:
                session.add_all(
                    ServiceAreaRow(city=a.city, country=a.country, active=True)
                    for a in _SEED
                )
                session.commit()

    def _refresh(self) -> None:
        with self._session_factory() as session:
            rows = session.scalars(
                select(ServiceAreaRow).where(ServiceAreaRow.active.is_(True))
            ).all()
        self._by_name = {r.city.lower(): ServiceArea(r.city, r.country) for r in rows}
        self._loaded_at = time.monotonic()

    def find(self, city_text: str) -> ServiceArea | None:
        if time.monotonic() - self._loaded_at > _CACHE_TTL_SECONDS:
            self._refresh()
        return _match(city_text, self._by_name)


_catalog: ServiceAreaCatalog | None = None


def configure(database_url: str) -> None:
    """Point the lookup at the database. Called once at startup."""
    global _catalog
    _catalog = ServiceAreaCatalog(database_url)


def find_service_area(city_text: str) -> ServiceArea | None:
    """Best-matching serviced location, from the database when configured."""
    if _catalog is not None:
        return _catalog.find(city_text)
    return _match(city_text, _SEED_BY_NAME)
