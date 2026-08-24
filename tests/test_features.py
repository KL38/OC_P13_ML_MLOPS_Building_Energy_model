"""The feature engineering contract.

`src/features.py` is the single source of truth shared by training and serving,
so what matters here is not that it computes something but that it computes the
*same* thing on both paths, and refuses anything it cannot encode faithfully.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src import features as F
from tests.conftest import RACINE


def test_le_schema_correspond_aux_artefacts():
    """The column order frozen in metadata.json is the one the code produces.

    A silent reorder would feed the model well-formed nonsense: every value in
    the wrong slot, no error raised anywhere.
    """
    metadata = json.loads((RACINE / "models" / "metadata.json").read_text("utf-8"))
    assert list(F.FEATURE_COLUMNS) == metadata["features"]


def test_trente_deux_variables(matrice: pd.DataFrame):
    assert matrice.shape[1] == 32
    assert list(matrice.columns) == list(F.FEATURE_COLUMNS)


def test_aucune_valeur_manquante(matrice: pd.DataFrame):
    assert not matrice.isna().to_numpy().any()


def test_les_deux_chemins_donnent_la_meme_ligne(description: pd.DataFrame):
    """Training path and serving path must agree, feature by feature.

    This is the test the whole design exists for: a building typed into the form
    has to traverse exactly the transformations a training row traverses. If it
    ever fails, the model is being served inputs it was not fitted on.
    """
    ligne = description.iloc[0]
    depuis_le_lot = F.build_features(description.head(1)).iloc[0]

    depuis_le_formulaire = F.build_features_one(
        year_built=int(ligne["YearBuilt"]),
        number_of_buildings=int(ligne["NumberofBuildings"]),
        number_of_floors=int(ligne["NumberofFloors"]),
        gfa_total=float(ligne["PropertyGFATotal"]),
        neighbourhood=ligne["Neighborhood"],
        usages=[
            (ligne[usage], ligne[gfa])
            for usage, gfa in F.USAGE_SLOTS
            if isinstance(ligne[usage], str) and ligne[gfa] > 0
        ],
        nb_usages=int(ligne["NbUsage"]),
        has_electricity=bool(ligne["HasElectricity"]),
        has_gas=bool(ligne["HasGas"]),
        has_steam=bool(ligne["HasSteam"]),
    ).iloc[0]

    pd.testing.assert_series_equal(
        depuis_le_lot, depuis_le_formulaire, check_names=False
    )


def test_la_composition_ne_depend_que_des_rapports():
    """Percentages and raw surfaces describing the same mix give the same row."""
    commun = {
        "year_built": 1980,
        "number_of_buildings": 1,
        "number_of_floors": 3,
        "gfa_total": 50_000,
        "neighbourhood": "DOWNTOWN",
    }
    en_pourcents = F.build_features_one(
        usages=[("Office", 70), ("Retail Store", 30)], **commun
    )
    en_surfaces = F.build_features_one(
        usages=[("Office", 35_000), ("Retail Store", 15_000)], **commun
    )
    pd.testing.assert_frame_equal(en_pourcents, en_surfaces)


def test_la_composition_somme_a_un(matrice: pd.DataFrame):
    parts = matrice[list(F.USAGE_GROUPS)].sum(axis=1)
    assert parts.between(0.999, 1.001).all()


def test_un_seul_quartier_actif(matrice: pd.DataFrame):
    colonnes = [c for c in F.FEATURE_COLUMNS if c.startswith("nb_")]
    assert (matrice[colonnes].sum(axis=1) == 1).all()


@pytest.mark.parametrize(
    "champ,valeur,motif",
    [
        ("neighbourhood", "PARIS 15E", "Unknown neighbourhood"),
        ("usages", [("Chateau", 100)], "Unknown property use type"),
    ],
)
def test_une_modalite_inconnue_leve(batiment: dict, champ, valeur, motif):
    """A mis-encoded building is worse than a refused prediction (E15)."""
    batiment[champ] = valeur
    with pytest.raises(ValueError, match=motif):
        F.build_features_one(**batiment)


def test_un_batiment_sans_usage_leve(batiment: dict):
    batiment["usages"] = []
    with pytest.raises(ValueError, match="at least one use type"):
        F.build_features_one(**batiment)


def test_une_surface_nulle_leve(batiment: dict):
    batiment["gfa_total"] = 0
    with pytest.raises(ValueError, match="strictly positive"):
        F.build_features_one(**batiment)


def test_les_colonnes_de_consommation_ne_survivent_pas(description: pd.DataFrame):
    """Their sum is the target: leaking them back in would be the P3 flaw (E08)."""
    assert not set(F.ENERGY_COLUMNS) & set(description.columns)
    assert {"HasElectricity", "HasGas", "HasSteam"} <= set(description.columns)


def test_le_batiment_hors_domaine_est_retire(description: pd.DataFrame):
    """Building 496 declares no usage breakdown at all -- out of domain (E14)."""
    assert len(description) == 1655
    assert 496 not in set(description["OSEBuildingID"])
