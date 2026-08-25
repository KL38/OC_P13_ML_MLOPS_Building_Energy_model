---
title: OC P13 Seattle building Predictor
emoji: 🏢
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.25.0
python_version: 3.12.12
app_file: app.py
pinned: false
short_description: Consommation et émissions estimées, avec leur fourchette
---

# Seattle Building Energy & Emissions

Estimate a building's annual energy use and greenhouse-gas emissions from its **structural
characteristics alone** — no meter reading required — and get every estimate with the interval
it deserves.

Seattle has required non-residential buildings over 20,000 sq ft to report their energy use
since 2010. This tool works on the buildings that *don't* report, so a sustainability team can
decide which ones to audit first.

## The three tabs

| Tab | What it does |
|---|---|
| **Bâtiment unique** | One building, typed into a form. Returns both estimates, their 75% intervals, and the SHAP factors driving them. |
| **Portefeuille** | Upload a CSV, get the fleet back sorted by decreasing emissions — the order in which to launch audits. |
| **Modèle & limites** | What the tool is for, and what it is not for. |

## What to expect

Two targets are predicted separately, because they answer different questions: energy use
locates the savings, emissions serve the carbon-neutrality objective. In Seattle they decouple —
the electricity is largely hydro, so a gas-heated building and an all-electric one with the same
consumption do not rank the same on carbon.

Every estimate ships with an interval calibrated by **conformal prediction** on errors actually
observed. On unseen buildings, that interval contains the true value 81% of the time for a 75%
target. It is roughly a factor of two wide — which is a property of the data, not a modelling
defect: among 163 comparable offices, energy use per square foot ranges from 27 to 91 kBtu.

> **The tool ranks, it does not price.** An estimate accurate to within a factor of two is usable
> for prioritising audits across a fleet. It is not usable for billing, contracting, or sizing an
> investment.

## Under the hood

GradientBoosting (energy, R² 0.745) and CatBoost (emissions, R² 0.723), trained on 1,655
buildings and 32 features, intervals from [MAPIE](https://mapie.readthedocs.io/)'s
`CrossConformalRegressor`. The interface is in French, its intended audience.

**Source code, test suite, CI/CD and the full project report:**
[github.com/KL38/OC_P13_ML_MLOPS_Building_Energy_model](https://github.com/KL38/OC_P13_ML_MLOPS_Building_Energy_model)
