"""Gradio app -- energy and emissions estimates for a Seattle building.

Three tabs: one building typed into a form, a portfolio uploaded as CSV, and a
page stating plainly what the model can and cannot do.

Presentation rules, all enforced in static/app.css rather than inline:

- Energy takes categorical slot 1 (blue), emissions slot 2 (orange), and keeps
  that identity across every block. UI chrome stays neutral so a button never
  impersonates a series.
- Each estimate leads with its value and carries a gauge of its interval. The
  marker sits at the estimate's true position inside the range, which is left of
  centre: the interval is symmetric in log space, not in kBtu.
- The coverage figures live in the "model and limits" tab, where they can be
  explained instead of asserted.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

# `spaces` doit etre importe avant gradio : ZeroGPU installe ses correctifs au
# moment de l'import, et un import tardif fait echouer le demarrage.
import spaces  # isort: skip

import gradio as gr
import numpy as np
import pandas as pd

from src import features as F
from src import model as M

RACINE = Path(__file__).resolve().parent


# ZeroGPU refuses to start a Space that declares no GPU function at all: it
# fails with "No @spaces.GPU function detected during startup" even when the
# app never needs a GPU. This placeholder satisfies that startup scan.
#
# It is never called, so no GPU is ever requested and the daily quota stays
# untouched -- the models are scikit-learn and CatBoost and run on the CPU
# allocation. The decorator is a no-op outside ZeroGPU, so local runs and CI
# are unaffected.
@spaces.GPU
def _presence_zerogpu() -> None:  # pragma: no cover
    """Never invoked. Exists only so ZeroGPU agrees to start the Space."""

MODELS, META = M.load()
CONFIDENCE = META["confidence_level"]
USAGE_TYPES = sorted(F.USAGE_MAP)

# Subject of the interval sentence, per target: it has to agree in French.
SUJET_FOURCHETTE = {
    "energie": "la vraie consommation se situe",
    "emissions": "les vraies émissions se situent",
}

# Plausibility bounds. Anything outside is a typing mistake rather than a
# building, and is refused with a message naming the offending field.
#
# Each message is written out instead of templated: French agreement differs
# between fields ("comprise" / "compris"), and a year must not carry thousands
# separators the way a surface does -- hence the trailing flag.
CONTROLES: dict[str, tuple[int, int, str, bool]] = {
    "annee": (
        1850,
        2016,
        "L'année de construction doit être comprise entre 1850 et 2016",
        False,
    ),
    "etages": (1, 120, "Le nombre d'étages doit être compris entre 1 et 120", False),
    "batiments": (
        1,
        100,
        "Le nombre de bâtiments doit être compris entre 1 et 100",
        False,
    ),
    "surface": (
        500,
        10_000_000,
        "La surface totale doit être comprise entre 500 et 10 000 000 pieds carrés",
        True,
    ),
}

# Beyond this the building is outside the range the model was fitted on. Not an
# error -- the estimate is still produced -- but the user is told.
SURFACE_HORS_DOMAINE = 2_000_000

# Columns a portfolio CSV must carry: the neutral description, not meter
# readings. Consumption columns are deliberately absent -- their sum is the
# target, so a user who has them does not need the model (E08).
PORTFOLIO_COLUMNS = [
    "YearBuilt",
    "NumberofBuildings",
    "NumberofFloors",
    "PropertyGFATotal",
    "Neighborhood",
    "NbUsage",
    "HasElectricity",
    "HasGas",
    "HasSteam",
    *(c for pair in F.USAGE_SLOTS for c in pair),
]

# The 32 features, folded into the five things a building owner would name.
# SHAP contributions are additive, so summing within a group is exact.
GROUPES_EXPLICATION: dict[str, tuple[str, ...]] = {
    "Surface et volumétrie": ("logGFAtotal", "NumberofFloors", "NumberofBuildings"),
    "Usages du bâtiment": (*F.USAGE_GROUPS, "NbUsage"),
    "Sources d'énergie": ("HasElectricity", "HasGas", "HasSteam"),
    "Quartier": tuple(c for c in F.FEATURE_COLUMNS if c.startswith("nb_")),
    "Année de construction": ("YearBuilt",),
}

# Every feature must land in exactly one group, or the explanation silently
# drops part of the prediction.
assert sorted(c for cols in GROUPES_EXPLICATION.values() for c in cols) == sorted(
    F.FEATURE_COLUMNS
)


# --------------------------------------------------------------- formatting


def _fr(value: float, decimals: int = 0) -> str:
    """Number with space thousands separators, French convention."""
    return f"{value:,.{decimals}f}".replace(",", " ").replace(".", ",")


def _decimales(target: str) -> int:
    return 0 if target == "energie" else 1


def _total_parts(part1, part2, part3) -> str:
    """Running total of the usage shares, refreshed as the user types."""
    total = sum(part or 0 for part in (part1, part2, part3))
    if abs(total - 100) <= 0.5:
        return f"**Total : {total:.0f} %**"
    return f"**Total : {total:.0f} %** — les parts doivent totaliser 100 %."


def _facteur(effet_log: float) -> str:
    """A log contribution said as a multiplier, the way a reader would say it."""
    facteur = float(np.exp(abs(effet_log)))
    if facteur < 1.05:
        return "sans effet"
    return f"×{_fr(facteur, 1)}" if effet_log > 0 else f"÷{_fr(facteur, 1)}"


def _infobulle(target: str) -> str:
    """Hover detail on the interval sentence: what the estimate is good for."""
    verbe = "émettre" if target == "emissions" else "consommer"
    lignes = (
        (
            f"Deux bâtiments aux caractéristiques identiques peuvent {verbe} "
            f"du simple au triple."
        ),
        (
            "Ce qui les sépare — taux d'occupation, horaires, équipements "
            "installés, âge et rendement des systèmes, isolation, consignes de "
            "chauffage — ne figure pas dans les données disponibles."
        ),
        "",
        (
            "L'estimation est donc indicative : elle sert à classer un parc et "
            "à cibler les audits, pas à chiffrer précisément un bâtiment."
        ),
    )
    # &#10; is the newline a browser honours inside a title attribute.
    return "&#10;".join(lignes)


# ------------------------------------------------------------ result blocks


def _html_resultat(target: str, row: pd.Series) -> str:
    """The estimate, then a gauge placing it inside its interval.

    A bare pair of bounds asks the reader to do the arithmetic. The gauge shows
    the span at a glance and puts the marker where the estimate actually falls
    -- about a third of the way in, because the interval is symmetric in log
    space and therefore lopsided in kBtu.
    """
    decimales = _decimales(target)
    estimation = row[f"{target}_estimation"]
    bas, haut = row[f"{target}_bas"], row[f"{target}_haut"]
    position = (estimation - bas) / (haut - bas) * 100.0
    facteur = (estimation / bas + haut / estimation) / 2.0

    return (
        f'<div class="oc oc--{target}">'
        '<div class="oc-result">'
        f'<div class="oc-result__label">{M.LABELS[target]}</div>'
        f'<div class="oc-result__value">{_fr(estimation, decimales)}'
        f'<span class="oc-result__unit">{M.UNITS[target]}</span></div>'
        '<div class="oc-gauge">'
        '<div class="oc-gauge__head"><span>Fourchette probable</span>'
        f'<span class="oc-pill">à un facteur {_fr(facteur, 1)} près</span></div>'
        '<div class="oc-gauge__track"><div class="oc-gauge__span"></div>'
        f'<div class="oc-gauge__marker" style="left:{position:.2f}%"></div></div>'
        f'<div class="oc-gauge__ends"><span>{_fr(bas, decimales)}</span>'
        f"<span>{_fr(haut, decimales)}</span></div></div>"
        f'<div class="oc-result__note">Notre modèle est confiant à '
        f"{CONFIDENCE * 100:.0f} % que {SUJET_FOURCHETTE[target]} dans cette "
        f'fourchette. <span class="oc-info" title="{_infobulle(target)}">'
        "&#9432;</span></div>"
        "</div></div>"
    )


def _html_facteurs(x: pd.DataFrame, target: str) -> str:
    """What multiplies or divides the estimate, in the app's own styling.

    The model predicts log1p of the target and SHAP decomposes that additively,
    so a sum in log space is a product in business units: each contribution *is*
    a multiplier. The log1p rather than log leaves a residual -- negligible on
    energy, about 1% on emissions -- and the reference point is the geometric
    mean of predictions, which is why it is labelled "référence".
    """
    _, values = M.explain(x, target)
    contributions = pd.Series(values, index=x.columns)
    effets = {
        nom: float(contributions[list(colonnes)].sum())
        for nom, colonnes in GROUPES_EXPLICATION.items()
    }

    ordre = sorted(effets, key=lambda nom: -abs(effets[nom]))
    limite = max(max(abs(v) for v in effets.values()), float(np.log(1.3))) * 1.18

    barres = []
    for nom in ordre:
        effet = effets[nom]
        part = abs(effet) / limite * 50.0
        sens = "up" if effet > 0 else "down"
        cote = "left" if effet > 0 else "right"
        barres.append(
            '<div class="oc-bar">'
            f'<div class="oc-bar__label">{nom}</div>'
            '<div class="oc-bar__plot"><div class="oc-bar__axis"></div>'
            f'<div class="oc-bar__fill oc-bar__fill--{sens}" '
            f'style="width:{part:.2f}%"></div>'
            f'<div class="oc-bar__value" '
            f'style="{cote}:calc(50% + {part:.2f}% + 9px)">{_facteur(effet)}</div>'
            "</div></div>"
        )

    graduations = [
        f for f in (1.5, 2, 3, 5, 10, 20) if limite * 0.18 < np.log(f) < limite * 0.94
    ]
    ticks = ['<span style="left:50%">référence</span>']
    for f in graduations:
        offset = float(np.log(f)) / limite * 50.0
        ticks.append(f'<span style="left:{50 + offset:.2f}%">×{f:g}</span>')
        ticks.append(f'<span style="left:{50 - offset:.2f}%">÷{f:g}</span>')

    return (
        f'<div class="oc oc--{target}">'
        f'<div class="oc-bars__title">{M.LABELS[target]}</div>'
        + "".join(barres)
        + '<div class="oc-scale"><div class="oc-scale__gutter"></div>'
        f'<div class="oc-scale__ticks">{"".join(ticks)}</div></div></div>'
    )


# ------------------------------------------------------- single building tab


def _valider(annee, nb_batiments, nb_etages, surface) -> None:
    """Refuse impossible inputs, warn on ones merely outside the training range."""
    saisies = (
        ("annee", annee),
        ("etages", nb_etages),
        ("batiments", nb_batiments),
        ("surface", surface),
    )
    for cle, valeur in saisies:
        mini, maxi, message, separateur = CONTROLES[cle]
        if valeur is None:
            raise gr.Error(f"{message} — champ non renseigné.")
        nombre = float(valeur)
        if not mini <= nombre <= maxi:
            saisie = _fr(nombre) if separateur else f"{nombre:.0f}"
            raise gr.Error(f"{message} — valeur saisie : {saisie}.")

    if float(surface) > SURFACE_HORS_DOMAINE:
        gr.Warning(
            "Cette surface dépasse celles du parc de Seattle sur lequel le modèle "
            "a été entraîné. L'estimation reste produite, mais sa fiabilité n'est "
            "plus garantie."
        )


def estimer(
    annee,
    nb_batiments,
    nb_etages,
    surface,
    quartier,
    usage1,
    part1,
    usage2,
    part2,
    usage3,
    part3,
    nb_usages,
    electricite,
    gaz,
    vapeur,
):
    """Form -> two result blocks and two factor charts."""
    _valider(annee, nb_batiments, nb_etages, surface)

    usages = [
        (use, share)
        for use, share in ((usage1, part1), (usage2, part2), (usage3, part3))
        if use and share and share > 0
    ]
    if not usages:
        raise gr.Error(
            "Renseignez au moins un usage avec une part strictement positive."
        )

    # usage_composition() normalises on the sum of the weights, which is right
    # for training data with inconsistent declared surfaces but would silently
    # reinterpret a form entry: 70/50/30 would be heard as 47/33/20. The
    # interface collects percentages, so it enforces percentages (E10).
    total = sum(share for _, share in usages)
    if abs(total - 100) > 0.5:
        raise gr.Error(
            f"Les parts d'usage doivent totaliser 100 % — actuellement {total:.0f} %."
        )

    x = F.build_features_one(
        year_built=int(annee),
        number_of_buildings=int(nb_batiments),
        number_of_floors=int(nb_etages),
        gfa_total=float(surface),
        neighbourhood=quartier,
        usages=usages,
        # The user may declare more usage types than the three the composition
        # holds; the count is a feature in its own right (E16).
        nb_usages=max(int(nb_usages), len(usages)),
        has_electricity=bool(electricite),
        has_gas=bool(gaz),
        has_steam=bool(vapeur),
    )
    row = M.predict(x).iloc[0]

    return (
        _html_resultat("energie", row),
        _html_resultat("emissions", row),
        _html_facteurs(x, "energie"),
        _html_facteurs(x, "emissions"),
        gr.update(visible=True),
    )


# ------------------------------------------------------------ portfolio tab


def _carte_kpi(libelle: str, valeur: str, unite: str) -> str:
    return (
        '<div class="oc-kpi">'
        f'<div class="oc-kpi__label">{libelle}</div>'
        f'<div class="oc-kpi__value">{valeur}'
        f'<span class="oc-kpi__unit">{unite}</span></div></div>'
    )


def _html_synthese(result: pd.DataFrame) -> str:
    """Portfolio-level reading: totals, concentration, dominant uses.

    The concentration bar is the point of the tab. A parc's emissions are almost
    never spread evenly, and knowing that a handful of buildings carry half of
    them is what turns a list into an audit plan.
    """
    nombre = len(result)
    surface = float(result["PropertyGFATotal"].sum())
    energie = float(result["energie_estimation"].sum())
    emissions = float(result["emissions_estimation"].sum())

    # result arrives sorted by estimated emissions, descending.
    parts = result["emissions_estimation"].to_numpy() / emissions
    cumul = np.cumsum(parts)
    seuil = int(np.searchsorted(cumul, 0.5) + 1)

    cartes = "".join(
        [
            _carte_kpi("Bâtiments", _fr(nombre), ""),
            _carte_kpi("Surface cumulée", _fr(surface / 1e6, 2), "M pi²"),
            _carte_kpi("Énergie estimée", _fr(energie / 1e6, 1), "M kBtu"),
            _carte_kpi("Émissions estimées", _fr(emissions), "tCO2e"),
            _carte_kpi("Intensité moyenne", _fr(energie / surface, 1), "kBtu/pi²"),
        ]
    )

    segments = "".join(
        [
            '<div class="oc-pareto__seg'
            + ("" if rang < seuil else " oc-pareto__seg--rest")
            + f'" style="width:{part * 100:.3f}%"></div>'
            for rang, part in enumerate(parts)
        ]
    )

    usages = (
        result.groupby("LargestPropertyUseType")["emissions_estimation"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
    )
    plus_grand = float(usages.iloc[0])
    lignes_usage = "".join(
        [
            '<div class="oc-share">'
            f'<div class="oc-share__label">{nom}</div>'
            '<div class="oc-share__track">'
            f'<div class="oc-share__fill" style="width:{v / plus_grand * 100:.2f}%">'
            "</div></div>"
            f'<div class="oc-share__value">{v / emissions * 100:.0f} %</div></div>'
            for nom, v in usages.items()
        ]
    )

    pluriel = "s" if seuil > 1 else ""
    return (
        '<div class="oc">'
        f'<div class="oc-kpis">{cartes}</div>'
        '<div class="oc-panel">'
        f'<div class="oc-panel__title"><b>{seuil} bâtiment{pluriel} sur {nombre}</b> '
        "concentrent la moitié des émissions estimées du parc — ce sont eux "
        "qu'un audit doit viser en premier.</div>"
        f'<div class="oc-pareto">{segments}</div>'
        '<div class="oc-panel__note">Chaque segment est un bâtiment, sa largeur '
        "sa part des émissions du parc.</div></div>"
        '<div class="oc-panel">'
        '<div class="oc-panel__title">Répartition des émissions par usage '
        "principal</div>"
        f"{lignes_usage}</div></div>"
    )


def traiter_portefeuille(fichier):
    """Portfolio CSV -> table sorted by estimated emissions, export, summary."""
    if fichier is None:
        raise gr.Error("Déposez un fichier CSV avant de lancer l'estimation.")

    try:
        source = pd.read_csv(fichier.name)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as erreur:
        raise gr.Error(f"Fichier CSV illisible : {erreur}") from None

    if source.empty:
        raise gr.Error("Le fichier ne contient aucune ligne.")

    missing = [c for c in PORTFOLIO_COLUMNS if c not in source.columns]
    if missing:
        raise gr.Error(f"Colonnes manquantes : {', '.join(missing)}")

    # An unknown neighbourhood or use type raises rather than being silently
    # mis-encoded -- the message names the offending value.
    try:
        X = F.build_features(source)
    except ValueError as erreur:
        raise gr.Error(str(erreur)) from None

    result = pd.concat([source, M.predict(X)], axis=1)
    result = result.sort_values("emissions_estimation", ascending=False)

    chemin = Path(tempfile.gettempdir()) / "estimations_portefeuille.csv"
    result.to_csv(chemin, index=False)

    # The downloaded file keeps raw numeric columns for downstream use. The table
    # on screen is for reading: each range collapses into one column, numbers get
    # French formatting, and emissions come first because they drive the order.
    def _plage_texte(bas, haut, decimales):
        return [f"{_fr(b, decimales)} – {_fr(h, decimales)}" for b, h in zip(bas, haut)]

    apercu = pd.DataFrame(
        {
            "Priorité": range(1, len(result) + 1),
            "Quartier": result["Neighborhood"].to_numpy(),
            "Usage principal": result["LargestPropertyUseType"].to_numpy(),
            "Surface (pi²)": [_fr(v) for v in result["PropertyGFATotal"]],
            "Émissions (tCO2e)": [_fr(v, 1) for v in result["emissions_estimation"]],
            "Fourchette émissions": _plage_texte(
                result["emissions_bas"], result["emissions_haut"], 1
            ),
            "Énergie (kBtu)": [_fr(v) for v in result["energie_estimation"]],
            "Fourchette énergie": _plage_texte(
                result["energie_bas"], result["energie_haut"], 0
            ),
        }
    )

    return apercu, str(chemin), _html_synthese(result), gr.update(visible=True)


# ----------------------------------------------------------- model & limits


def _page_limites() -> str:
    lignes = "\n".join(
        f"| {M.LABELS[cible]} | {spec['estimateur']} | {spec['R2_log_hors_fold']:.3f} "
        f"| {spec['MedAPE_hors_fold']:.1%} | {spec['couverture_observee_test']:.1%} "
        f"| facteur {spec['facteur_intervalle']:.2f} |"
        for cible, spec in META["modeles"].items()
    )
    return f"""
## Ce que valent ces estimations

| Cible | Modèle | R² (log) | Erreur médiane | Couverture observée | Largeur de fourchette |
|---|---|---|---|---|---|
{lignes}

Entraînement sur **{META["n_batiments_entrainement"]} bâtiments** du benchmark
{META["jeu_source"]}, à partir de 32 variables structurelles.

## Comment lire la fourchette

Les fourchettes sont produites par **prédiction conforme** (MAPIE,
`CrossConformalRegressor`). Elles sont calibrées pour contenir la vraie valeur
dans **{CONFIDENCE * 100:.0f} % des cas** : sur un parc de 1 000 bâtiments
estimés de cette façon, environ {CONFIDENCE * 1000:.0f} verront leur
consommation réelle tomber dans la fourchette affichée.

Cette calibration porte sur **l'ensemble, pas sur un bâtiment isolé**. Elle ne
se transporte pas non plus automatiquement à chaque typologie : un type de
bâtiment peut être couvert à 85 % et un autre à 68 %, la moyenne restant à
{CONFIDENCE * 100:.0f} %. C'est la limite principale de la méthode.

## Pourquoi la fourchette est large

Elle ne l'est pas par défaut de modélisation : **c'est l'incertitude réelle du
problème**. Parmi 163 immeubles de bureaux de taille comparable, la consommation
au pied carré va de 27 à 91 kBtu — un facteur 3,4 que rien dans les données ne
permet de départager. Une fourchette construite sur cette seule dispersion
vaudrait un facteur 2,24 ; le modèle produit un facteur 2,03. **Il est déjà plus
serré que l'écart naturel entre bâtiments indiscernables.**

La raison est physique : les variables disponibles décrivent **ce qu'un bâtiment
est**, alors que sa consommation dépend de **comment il est utilisé et exploité** —
taux d'occupation et horaires, équipements installés, âge et rendement des
systèmes, qualité de l'enveloppe, consignes de température. Rien de tout cela ne
figure dans le benchmark de Seattle.

## Ce que le modèle ne sait pas faire

- **Sortir de Seattle.** Un autre climat, un autre parc, une autre réglementation
  cassent l'hypothèse d'échangeabilité sur laquelle repose la calibration.
- **Facturer, contractualiser ou dimensionner un investissement.** À un facteur 2
  près, la précision est celle d'un outil de priorisation, pas d'un instrument
  de mesure.
- **Remplacer un relevé.** L'estimation classe un parc et cible les audits ;
  elle ne se substitue pas à un compteur.
- **Traiter un bâtiment à énergie positive.** Le seul cas du parc — le Bullitt
  Center, qui réinjecte plus d'électricité qu'il n'en consomme — a été retiré du
  jeu d'entraînement ; le modèle ne sait pas produire une valeur négative.

## Empreinte de l'entraînement

Le benchmark complet — 12 modèles, 2 protocoles — a consommé **7,3 Wh** et émis
**0,41 g de CO₂eq** (CodeCarbon, mix électrique français). Le modèle de
fondation TabPFN, écarté, en représentait à lui seul 84 % pour 2 points de R²
dans le bruit d'échantillonnage.
"""


# ------------------------------------------------------------------ layout

# Neutral chrome: Gradio's default accent is orange, which is the emissions
# series colour. A slate primary keeps buttons legible as controls rather than
# as data.
# The system sans rather than Gradio's Montserrat: a geometric display face
# reads as decoration on a measurement tool, and the system stack costs no
# network request. Tighter radii for the same reason -- pill-shaped labels look
# playful where this interface wants to look sober.
THEME = gr.themes.Soft(
    primary_hue="slate",
    neutral_hue="slate",
    radius_size="sm",
    font=["system-ui", "-apple-system", "Segoe UI", "Helvetica Neue", "sans-serif"],
    font_mono=["ui-monospace", "SFMono-Regular", "Consolas", "monospace"],
)

with gr.Blocks(title="Estimation énergétique — bâtiments de Seattle") as demo:
    gr.Markdown(
        "# Estimation énergétique d'un bâtiment\n"
        "Consommation et émissions estimées **à partir des seules caractéristiques "
        "structurelles**, sans aucun relevé de compteur."
    )

    with gr.Tabs():
        with gr.Tab("Bâtiment unique"):
            # Entry order follows how someone describes a building out loud:
            # where and how big, then what it is used for, then how it is heated.
            with gr.Row():
                with gr.Column():
                    gr.Markdown("LE BÂTIMENT", elem_classes="oc-section")
                    quartier = gr.Dropdown(
                        label="Quartier",
                        choices=list(F.NEIGHBOURHOODS),
                        value="DOWNTOWN",
                    )
                    surface = gr.Number(
                        label="Surface totale (pieds carrés)",
                        value=50000,
                        minimum=CONTROLES["surface"][0],
                        maximum=CONTROLES["surface"][1],
                    )
                    annee = gr.Number(
                        label="Année de construction",
                        value=1980,
                        minimum=CONTROLES["annee"][0],
                        maximum=CONTROLES["annee"][1],
                        precision=0,
                    )
                    with gr.Row():
                        nb_etages = gr.Number(
                            label="Étages",
                            value=3,
                            minimum=CONTROLES["etages"][0],
                            maximum=CONTROLES["etages"][1],
                            precision=0,
                        )
                        nb_batiments = gr.Number(
                            label="Bâtiments",
                            value=1,
                            minimum=CONTROLES["batiments"][0],
                            maximum=CONTROLES["batiments"][1],
                            precision=0,
                        )

                    gr.Markdown("SES SOURCES D'ÉNERGIE", elem_classes="oc-section")
                    electricite = gr.Checkbox(label="Électricité", value=True)
                    gaz = gr.Checkbox(label="Gaz naturel", value=False)
                    vapeur = gr.Checkbox(label="Réseau de vapeur", value=False)

                with gr.Column():
                    gr.Markdown("SES USAGES", elem_classes="oc-section")
                    gr.Markdown(
                        "Les trois usages principaux et leur part de surface, "
                        "en pourcentage. Les parts doivent faire 100 %."
                    )
                    with gr.Row():
                        usage1 = gr.Dropdown(
                            label="Usage principal",
                            choices=USAGE_TYPES,
                            value="Office",
                            scale=3,
                        )
                        part1 = gr.Number(
                            label="Part (%)", value=100, minimum=0, maximum=100, scale=1
                        )
                    with gr.Row():
                        usage2 = gr.Dropdown(
                            label="Usage secondaire",
                            choices=USAGE_TYPES,
                            value=None,
                            scale=3,
                        )
                        part2 = gr.Number(
                            label="Part (%)", value=0, minimum=0, maximum=100, scale=1
                        )
                    with gr.Row():
                        usage3 = gr.Dropdown(
                            label="Usage tertiaire",
                            choices=USAGE_TYPES,
                            value=None,
                            scale=3,
                        )
                        part3 = gr.Number(
                            label="Part (%)", value=0, minimum=0, maximum=100, scale=1
                        )

                    total_parts = gr.Markdown(_total_parts(100, 0, 0))
                    for champ in (part1, part2, part3):
                        champ.change(
                            _total_parts,
                            inputs=[part1, part2, part3],
                            outputs=total_parts,
                        )

                    nb_usages = gr.Number(
                        label="Nombre total d'usages déclarés",
                        info="Si le bâtiment en abrite plus de trois, indiquez-le "
                        "ici : le compte est une variable à part entière.",
                        value=1,
                        minimum=1,
                        precision=0,
                    )


            bouton = gr.Button(
                "Estimer",
                variant="primary",
                size="lg",
                elem_id="oc-estimer",
                elem_classes="oc-cta",
            )

            # Results stay hidden until there is something to show, so the page
            # does not open on two empty frames.
            with gr.Column(visible=False) as bloc_resultats:
                gr.Markdown("---")
                with gr.Row():
                    resultat_energie = gr.HTML()
                    resultat_emissions = gr.HTML()

                gr.Markdown("---")
                gr.Markdown(
                    """#### Facteurs déterminants de la consommation et des émissions

Chaque caractéristique multiplie ou divise l'estimation par rapport à un
bâtiment médian du parc de Seattle."""
                )
                with gr.Row():
                    facteurs_energie = gr.HTML()
                    facteurs_emissions = gr.HTML()

            bouton.click(
                estimer,
                inputs=[
                    annee,
                    nb_batiments,
                    nb_etages,
                    surface,
                    quartier,
                    usage1,
                    part1,
                    usage2,
                    part2,
                    usage3,
                    part3,
                    nb_usages,
                    electricite,
                    gaz,
                    vapeur,
                ],
                outputs=[
                    resultat_energie,
                    resultat_emissions,
                    facteurs_energie,
                    facteurs_emissions,
                    bloc_resultats,
                ],
                api_name="estimer",
                show_progress="full",
            )

        with gr.Tab("Portefeuille"):
            gr.Markdown(
                "## Estimer un parc entier\n"
                "Déposez un CSV décrivant vos bâtiments : le tableau ressort trié "
                "par émissions estimées décroissantes — l'ordre dans lequel lancer "
                "les audits.\n\n"
                "**Colonnes attendues :** `" + "`, `".join(PORTFOLIO_COLUMNS) + "`\n\n"
                "Aucune colonne de consommation : c'est précisément ce que le "
                "modèle estime."
            )
            fichier = gr.File(label="Fichier CSV", file_types=[".csv"])
            gr.Markdown(
                "Pas de fichier sous la main ? `data/exemple_portefeuille.csv` "
                "du dépôt contient dix bâtiments réels du parc de Seattle."
            )
            lancer = gr.Button(
                "Estimer le portefeuille", variant="primary", elem_classes="oc-cta"
            )

            with gr.Column(visible=False) as bloc_portefeuille:
                tableau = gr.Dataframe(label="Résultats", wrap=True)
                telechargement = gr.File(label="Télécharger les résultats complets")
                gr.Markdown("---")
                gr.Markdown("#### Lecture d'ensemble du portefeuille")
                synthese = gr.HTML()

            lancer.click(
                traiter_portefeuille,
                inputs=[fichier],
                outputs=[tableau, telechargement, synthese, bloc_portefeuille],
                api_name="portefeuille",
                show_progress="full",
            )

        with gr.Tab("Modèle & limites"):
            gr.Markdown(_page_limites())


if __name__ == "__main__":
    demo.launch(theme=THEME, css_paths=[str(RACINE / "static" / "app.css")])
