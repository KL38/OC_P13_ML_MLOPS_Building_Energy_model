"""The interface layer: what it refuses, and what it produces.

Everything here goes through the same entry points the buttons call, so a test
passing is evidence the app works and not merely that its helpers do.
"""

from __future__ import annotations

import gradio as gr
import pytest

import app
from tests.conftest import RACINE

EXEMPLE = RACINE / "data" / "exemple_portefeuille.csv"


class FichierDepose:
    """What gr.File hands the handler: an object carrying a path."""

    def __init__(self, chemin):
        self.name = str(chemin)


def test_le_marqueur_zerogpu_existe():
    """ZeroGPU refuses to start the Space without a decorated function.

    Deleting it as dead code would break the deployment and nothing else, so
    the guard belongs here rather than in a comment.
    """
    assert callable(app._presence_zerogpu)


# ------------------------------------------------------------- validation


def test_un_batiment_plausible_passe(batiment: dict):
    app._valider(1980, 1, 3, 50_000)


@pytest.mark.parametrize(
    "annee,batiments,etages,surface,attendu",
    [
        (2050, 1, 3, 50_000, "année de construction"),
        (1980, 1, 900, 50_000, "nombre d'étages"),
        (1980, 500, 3, 50_000, "nombre de bâtiments"),
        (1980, 1, 3, 99_000_000, "surface totale"),
        (None, 1, 3, 50_000, "année de construction"),
    ],
)
def test_une_valeur_aberrante_est_refusee(annee, batiments, etages, surface, attendu):
    """The message must name the offending field, not just say 'invalid'."""
    with pytest.raises(gr.Error) as capture:
        app._valider(annee, batiments, etages, surface)
    assert attendu in capture.value.message


def test_une_surface_hors_domaine_avertit_sans_bloquer():
    """Beyond the training range the estimate is still produced, with a warning."""
    with pytest.warns(UserWarning, match="parc de Seattle"):
        app._valider(1980, 1, 3, 5_000_000)


# ------------------------------------------------- single-building estimate


def test_l_estimation_produit_quatre_blocs_et_les_revele():
    sorties = app.estimer(
        1980,
        1,
        3,
        50_000,
        "DOWNTOWN",
        "Office",
        70,
        "Retail Store",
        20,
        "Parking",
        10,
        5,
        True,
        True,
        False,
    )
    resultat_energie, resultat_emissions, facteurs_energie, facteurs_emissions, bloc = (
        sorties
    )

    assert 'class="oc oc--energie"' in resultat_energie
    assert 'class="oc oc--emissions"' in resultat_emissions
    assert "oc-gauge__marker" in resultat_energie
    assert "oc-bar__fill" in facteurs_energie
    assert facteurs_energie != facteurs_emissions  # two models, two explanations
    assert bloc["visible"] is True


def test_les_parts_doivent_totaliser_cent():
    with pytest.raises(gr.Error, match="100 %"):
        app.estimer(
            1980,
            1,
            3,
            50_000,
            "DOWNTOWN",
            "Office",
            70,
            "Retail Store",
            50,
            "Parking",
            30,
            3,
            True,
            True,
            False,
        )


def test_au_moins_un_usage_est_exige():
    with pytest.raises(gr.Error, match="au moins un usage"):
        app.estimer(
            1980,
            1,
            3,
            50_000,
            "DOWNTOWN",
            None,
            0,
            None,
            0,
            None,
            0,
            1,
            True,
            True,
            False,
        )


def test_le_total_des_parts_est_affiche():
    assert "100 %" in app._total_parts(70, 20, 10)
    assert "doivent totaliser" in app._total_parts(70, 50, 30)
    assert "doivent totaliser" not in app._total_parts(100, 0, 0)


# --------------------------------------------------------------- portfolio


def test_le_portefeuille_exemple_passe_de_bout_en_bout():
    apercu, chemin, synthese, bloc = app.traiter_portefeuille(FichierDepose(EXEMPLE))

    assert len(apercu) == 10
    assert list(apercu["Priorité"]) == list(range(1, 11))
    assert "oc-pareto" in synthese
    assert "concentrent la moitié" in synthese
    assert bloc["visible"] is True

    import pandas as pd

    export = pd.read_csv(chemin)
    assert len(export) == 10
    for cible in ("energie", "emissions"):
        assert (export[f"{cible}_bas"] <= export[f"{cible}_estimation"]).all()
        assert (export[f"{cible}_estimation"] <= export[f"{cible}_haut"]).all()
    # Sorted by estimated emissions, descending: that ordering is the deliverable.
    assert export["emissions_estimation"].is_monotonic_decreasing


def test_un_fichier_absent_est_refuse():
    with pytest.raises(gr.Error, match="Déposez un fichier"):
        app.traiter_portefeuille(None)


def test_une_colonne_manquante_est_nommee(tmp_path):
    import pandas as pd

    tronque = pd.read_csv(EXEMPLE).drop(columns=["YearBuilt"])
    chemin = tmp_path / "tronque.csv"
    tronque.to_csv(chemin, index=False)

    with pytest.raises(gr.Error, match="YearBuilt"):
        app.traiter_portefeuille(FichierDepose(chemin))


def test_le_fichier_exemple_porte_exactement_le_gabarit():
    """It doubles as the reference template shown in the tab."""
    import pandas as pd

    assert list(pd.read_csv(EXEMPLE).columns) == app.PORTFOLIO_COLUMNS
