"""The serving contract of the exported artefacts.

Two kinds of check live here. The invariants -- ordered bounds, positive values,
SHAP additivity -- must hold for any artefact. The non-regression thresholds
catch a broken export or a feature-engineering change that silently degrades the
model, and are deliberately loose.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import features as F
from src import model as M
from tests.conftest import CIBLES

# In-sample, therefore optimistic: these numbers do NOT measure performance --
# the honest figures are the out-of-fold ones in metadata.json. They exist to
# fail loudly if an artefact or the feature engineering breaks. Measured at
# 0.310 / 0.285; the ceiling leaves room for retraining noise.
MEDAPE_MAXIMAL = {"energie": 0.40, "emissions": 0.40}
COUVERTURE_MINIMALE = 0.75


def test_les_deux_artefacts_se_chargent(metadonnees: dict):
    modeles, _ = M.load()
    assert set(modeles) == {"energie", "emissions"}
    assert set(metadonnees["modeles"]) == {"energie", "emissions"}


def test_le_chargement_est_mis_en_cache():
    assert M.load() is M.load()


def test_la_sortie_a_les_colonnes_attendues(predictions: pd.DataFrame):
    attendues = {
        f"{cible}_{suffixe}"
        for cible in CIBLES
        for suffixe in ("estimation", "bas", "haut")
    }
    assert set(predictions.columns) == attendues


@pytest.mark.parametrize("cible", list(CIBLES))
def test_les_bornes_encadrent_l_estimation(predictions: pd.DataFrame, cible: str):
    """lower <= estimate <= upper, on every building of the dataset."""
    bas = predictions[f"{cible}_bas"]
    estimation = predictions[f"{cible}_estimation"]
    haut = predictions[f"{cible}_haut"]
    assert (bas <= estimation).all()
    assert (estimation <= haut).all()


@pytest.mark.parametrize("cible", list(CIBLES))
def test_aucune_borne_negative(predictions: pd.DataFrame, cible: str):
    """A negative lower bound is physically meaningless and must never surface.

    It is not hypothetical: on a log1p target, a building estimated around
    1 tCO2e gets a lower bound below zero before clamping.
    """
    assert (predictions[f"{cible}_bas"] >= 0).all()
    assert (predictions[f"{cible}_estimation"] > 0).all()


@pytest.mark.parametrize("cible", list(CIBLES))
def test_la_largeur_reste_proche_de_la_reference(
    predictions: pd.DataFrame, metadonnees: dict, cible: str
):
    facteur = np.median(
        predictions[f"{cible}_haut"] / predictions[f"{cible}_estimation"]
    )
    attendu = metadonnees["modeles"][cible]["facteur_intervalle"]
    assert facteur == pytest.approx(attendu, rel=0.05)


@pytest.mark.parametrize("cible", list(CIBLES))
def test_shap_reconstruit_exactement_la_prediction(matrice: pd.DataFrame, cible: str):
    """base + sum(contributions) == the log prediction, to floating point.

    It holds because `predict` is the plain mean of the five fold models and the
    explanation averages those same five. Explaining `single_estimator_` instead
    would drift by up to 0.23 in log space.
    """
    x = matrice.head(1)
    base, valeurs = M.explain(x, cible)
    modeles, _ = M.load()
    assert base + valeurs.sum() == pytest.approx(modeles[cible].predict(x)[0], abs=1e-9)


@pytest.mark.parametrize("cible", list(CIBLES))
def test_shap_couvre_toutes_les_variables(matrice: pd.DataFrame, cible: str):
    _, valeurs = M.explain(matrice.head(1), cible)
    assert len(valeurs) == len(F.FEATURE_COLUMNS)


@pytest.mark.parametrize("cible", list(CIBLES))
def test_non_regression_erreur_mediane(
    description: pd.DataFrame, predictions: pd.DataFrame, cible: str
):
    reel = description[CIBLES[cible]].to_numpy()
    estime = predictions[f"{cible}_estimation"].to_numpy()
    medape = float(np.median(np.abs(estime - reel) / reel))
    assert medape <= MEDAPE_MAXIMAL[cible]


@pytest.mark.parametrize("cible", list(CIBLES))
def test_non_regression_couverture(
    description: pd.DataFrame, predictions: pd.DataFrame, cible: str
):
    reel = description[CIBLES[cible]].to_numpy()
    dedans = (reel >= predictions[f"{cible}_bas"]) & (
        reel <= predictions[f"{cible}_haut"]
    )
    assert float(dedans.mean()) >= COUVERTURE_MINIMALE
