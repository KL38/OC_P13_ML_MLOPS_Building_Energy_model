"""Shared fixtures.

The heavy objects -- the two MAPIE artefacts and the cleaned dataset -- are
session-scoped: loading them once keeps the suite under a couple of seconds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

from src import features as F
from src import model as M

CIBLES = {"energie": "SiteEnergyUse(kBtu)", "emissions": "TotalGHGEmissions"}


@pytest.fixture(scope="session")
def description() -> pd.DataFrame:
    """The cleaned benchmark, turned into the neutral description."""
    return F.from_benchmark(pd.read_csv(RACINE / "data" / "dfclean.csv"))


@pytest.fixture(scope="session")
def matrice(description: pd.DataFrame) -> pd.DataFrame:
    """The full feature matrix, in FEATURE_COLUMNS order."""
    return F.build_features(description)


@pytest.fixture(scope="session")
def predictions(matrice: pd.DataFrame) -> pd.DataFrame:
    """Estimates and interval bounds for every building of the dataset."""
    return M.predict(matrice)


@pytest.fixture(scope="session")
def metadonnees() -> dict:
    return M.load()[1]


@pytest.fixture
def batiment() -> dict:
    """A plausible mid-sized office, as the form would collect it."""
    return {
        "year_built": 1980,
        "number_of_buildings": 1,
        "number_of_floors": 3,
        "gfa_total": 50_000,
        "neighbourhood": "DOWNTOWN",
        "usages": [("Office", 70), ("Retail Store", 20), ("Parking", 10)],
        "nb_usages": 5,
        "has_electricity": True,
        "has_gas": True,
        "has_steam": False,
    }
