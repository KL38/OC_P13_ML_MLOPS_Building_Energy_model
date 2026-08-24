---
title: Estimation énergétique des bâtiments de Seattle
emoji: 🏢
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.25.0
python_version: 3.12.12
app_file: app.py
pinned: false
short_description: Consommation et émissions estimées, avec intervalles de confiance
---

# Estimation énergétique des bâtiments de Seattle

Mise en production du modèle de prédiction énergétique développé en projet 3
(*Seattle Building Energy Benchmarking 2016*). L'outil estime la **consommation
d'énergie** et les **émissions de gaz à effet de serre** d'un bâtiment à partir
de ses seules caractéristiques structurelles — sans le moindre relevé de
compteur — et accompagne chaque estimation de son **intervalle de confiance**.

**[→ Application en ligne](https://huggingface.co/spaces/KLEB38/OC_P13_seattle_energy_emission_predictions)**

## Le modèle en un tableau

| Cible | Modèle | R² (log) | Erreur médiane | Couverture observée | Intervalle |
|---|---|---|---|---|---|
| Énergie | GradientBoosting | 0,745 | 36,6 % | 81,3 % | ×÷2,03 |
| Émissions | CatBoost | 0,723 | 45,1 % | 81,0 % | ×÷2,34 |

Entraînement sur 1 655 bâtiments, 32 variables, cible en `log1p`. Intervalles
par **prédiction conforme** (MAPIE `CrossConformalRegressor`, méthode CV+),
niveau visé 75 %.

## Ce que le projet démontre

**Un modèle utilisable, pas un modèle flatteur.** P3 atteignait 0,81 de R² sur
les émissions grâce à des ratios de sources d'énergie calculés depuis la
consommation réelle — un utilisateur qui les connaît connaît déjà la réponse.
Ces ratios sont remplacés par de simples booléens de présence. La performance
mesurée baisse de 10 points ; le modèle devient utilisable sur un bâtiment non
relevé, ce qui est le cas d'usage.

**Une incertitude quantifiée plutôt que masquée.** Un nombre seul ne dit rien de
sa fiabilité. Chaque estimation sort avec un intervalle dont la couverture est
garantie sans hypothèse sur la loi des erreurs.

**Un plafond de performance démontré.** L'intervalle large n'est pas un défaut
de modélisation : parmi 163 immeubles de bureaux de taille comparable, la
consommation au pied carré varie d'un facteur 3,4. Un intervalle construit sur
cette seule dispersion irréductible vaudrait ×÷2,24 — le modèle produit ×÷2,03.
Il est déjà plus serré que l'écart naturel entre bâtiments indiscernables.

**Des arbitrages mesurés, pas opinés.** `ParkingRatio`, le filtrage des
bâtiments atypiques et `ConformalizedQuantileRegressor` ont été testés puis
écartés sur la base de mesures. TabPFN, meilleur score du benchmark, est écarté
pour deux motifs indépendants : sa licence interdit la production, et il
consomme 101 fois plus d'énergie que le modèle retenu pour 2 points de R² dans
le bruit d'échantillonnage.

## Architecture

```
Entraînement (local)                    Serving (Hugging Face)
────────────────────                    ──────────────────────
notebooks/03_modelisation.ipynb         Space Gradio (ZeroGPU)
  ├── MLflow (SQLite) ──── traçabilité   ├── app.py
  ├── CodeCarbon ───────── empreinte     ├── src/features.py  (partagé)
  └── MAPIE ────────────── models/ ────> └── src/model.py
                              ▲
                    GitHub Actions (ruff → pytest → upload_folder)
```

`src/features.py` est la **source unique** du feature engineering : un bâtiment
saisi dans l'interface traverse exactement les mêmes transformations qu'une
ligne du jeu d'entraînement. Aucun état appris n'y est persisté — les
vocabulaires sont des constantes versionnées avec le code, et une modalité
inconnue lève une exception plutôt que de produire un encodage silencieusement
faux.

Les artefacts (2,1 Mo) sont versionnés dans `models/` plutôt que tirés du Hub au
démarrage : à cette taille, un téléchargement n'ajouterait qu'un mode de panne.

## Structure

```
OC_P13/
├── app.py                  # interface Gradio (3 onglets)
├── requirements.txt        # dépendances du Space, versions épinglées
├── src/
│   ├── features.py         # feature engineering partagé entraînement/inférence
│   └── model.py            # chargement, prédiction, explication SHAP
├── models/                 # artefacts MAPIE + metadata.json
├── notebooks/
│   ├── 01_EDA.ipynb        # nettoyage → data/dfclean.csv
│   ├── 02_bivariee.ipynb   # analyse bivariée, validation du FE
│   └── 03_modelisation.ipynb  # benchmark, conformalisation, export
├── data/
└── docs/
```

## Lancer en local

```powershell
uv sync
uv run python app.py
```

## Documentation

| Document | Contenu |
|---|---|
| [`docs/ecarts_vs_P3.md`](docs/ecarts_vs_P3.md) | Journal des 16 divergences avec P3, une ligne par changement, plus les écarts assumés |
| [`docs/plafond_de_performance.md`](docs/plafond_de_performance.md) | Pourquoi ×÷2 est la bonne réponse et non un aveu de faiblesse |
| [`docs/note_conformal_prediction.md`](docs/note_conformal_prediction.md) | La prédiction conforme expliquée, avec exemples chiffrés |
| [`docs/empreinte_carbone.md`](docs/empreinte_carbone.md) | Ce que la mesure CodeCarbon a réellement tranché, et ses limites |

## Limites

L'outil sert à **classer un parc et prioriser des audits**. Il ne sert pas à
facturer, contractualiser ou dimensionner un investissement. La garantie de
couverture est *marginale* — vraie en moyenne, pas typologie par typologie — et
suppose des bâtiments comparables à ceux de Seattle.
