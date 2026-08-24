"""Deployment configuration must not drift from what is actually installed.

The classic failure of a Space is silent: a library gets bumped locally, the
artefacts are re-exported against it, and `requirements.txt` still pins the old
version. Nothing errors -- the Space simply serves a model unpickled by a
different library than the one that fitted it. These tests make that drift fail
in CI instead.
"""

from __future__ import annotations

import importlib.metadata as metadata_lib
import json
import re

import pytest

from tests.conftest import RACINE

# ZeroGPU accepts these two Python versions and no others.
PYTHON_ZEROGPU = {"3.12.12", "3.10.13"}

README = (RACINE / "README.md").read_text(encoding="utf-8")
REQUIREMENTS = (RACINE / "requirements.txt").read_text(encoding="utf-8")


def _entete_yaml() -> dict[str, str]:
    """The Space configuration block at the top of README.md."""
    bloc = README.split("---")[1]
    return dict(re.findall(r"^(\w+):\s*(.+?)\s*$", bloc, flags=re.MULTILINE))


def _versions_epinglees() -> dict[str, str]:
    return dict(
        re.findall(
            r"^([A-Za-z0-9_.-]+)==([\d.]+)\s*$", REQUIREMENTS, flags=re.MULTILINE
        )
    )


def test_le_space_declare_le_sdk_gradio():
    entete = _entete_yaml()
    assert entete["sdk"] == "gradio"
    assert entete["app_file"] == "app.py"


def test_la_version_du_sdk_suit_gradio_installe():
    assert _entete_yaml()["sdk_version"] == metadata_lib.version("gradio")


def test_la_version_de_python_est_acceptee_par_zerogpu():
    assert _entete_yaml()["python_version"] in PYTHON_ZEROGPU


@pytest.mark.parametrize(
    "paquet",
    [
        "gradio",
        "scikit-learn",
        "catboost",
        "mapie",
        "shap",
        "pandas",
        "numpy",
        "joblib",
    ],
)
def test_les_epinglages_correspondent_a_l_installe(paquet: str):
    epingles = _versions_epinglees()
    assert paquet in epingles, f"{paquet} absent de requirements.txt"
    assert epingles[paquet] == metadata_lib.version(paquet)


def test_les_epinglages_correspondent_aux_artefacts():
    """The libraries that fitted the models are the ones the Space will install."""
    metadata = json.loads((RACINE / "models" / "metadata.json").read_text("utf-8"))
    epingles = _versions_epinglees()
    for paquet, version in metadata["versions"].items():
        assert epingles[paquet] == version, (
            f"{paquet} : artefact construit en {version}, "
            f"Space epinglé en {epingles[paquet]}"
        )


def test_mlflow_ne_part_pas_dans_le_space():
    """Training-only tooling has no business in the serving image."""
    epingles = _versions_epinglees()
    for absent in ("mlflow", "codecarbon", "tabpfn"):
        assert absent not in epingles


def test_la_feuille_de_style_existe():
    assert (RACINE / "static" / "app.css").is_file()
