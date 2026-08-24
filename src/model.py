"""Model loading, prediction and explanation for the Gradio app.

The artefacts are ``CrossConformalRegressor`` objects fitted on ``log1p`` of each
target, so every number they produce -- point estimate and both bounds -- goes
through ``expm1`` before it reaches a user. That is valid because ``expm1`` is
monotonic: the coverage guarantee crosses the transformation intact.

MAPIE 1.5 exposes no public handle on its fitted sub-estimators, so the SHAP path
reaches into a private attribute. The intrusion is deliberate and was verified:
``predict`` returns the plain mean of the five fold models (equal to 1e-15), so
averaging their five SHAP explanations reconstructs exactly the prediction shown.
Explaining ``single_estimator_`` instead would describe a different model -- it
drifts up to 0.23 in log space, about 26% once exponentiated.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Business units of each target, for labelling. Seattle publishes site energy in
# kBtu and greenhouse gases in metric tons of CO2 equivalent.
UNITS: dict[str, str] = {"energie": "kBtu", "emissions": "tCO2e"}
LABELS: dict[str, str] = {
    "energie": "Consommation d'énergie",
    "emissions": "Émissions de gaz à effet de serre",
}


@lru_cache(maxsize=1)
def load() -> tuple[dict, dict]:
    """(models by target, metadata). Read once, then served from cache."""
    metadata = json.loads((MODELS_DIR / "metadata.json").read_text(encoding="utf-8"))
    models = {
        target: joblib.load(MODELS_DIR / spec["fichier"])
        for target, spec in metadata["modeles"].items()
    }
    return models, metadata


def predict(X: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix -> point estimate and interval bounds, in business units.

    One triplet of columns per target, named ``<target>_estimation``,
    ``<target>_bas`` and ``<target>_haut``.
    """
    models, _ = load()
    out = pd.DataFrame(index=X.index)

    for target, model in models.items():
        y_log, intervals = model.predict_interval(X)
        out[f"{target}_estimation"] = np.expm1(y_log)
        # Clamped at zero. On a very small target, log1p(y) is close to 0 and
        # subtracting the conformity quantile pushes the lower bound negative --
        # one building of the 1655 lands at -0.13 tCO2e.
        #
        # This costs no coverage *within the model's domain*, which the EDA
        # restricts to strictly positive consumption: the one Seattle building
        # that exports more electricity than it draws, the Bullitt Center
        # (49784, -115 417 kBtu, -0.8 tCO2e), is removed there as a lone case.
        # A net-positive building is therefore out of domain, and no target the
        # model was fitted on can be negative.
        out[f"{target}_bas"] = np.expm1(intervals[:, 0, 0]).clip(min=0.0)
        out[f"{target}_haut"] = np.expm1(intervals[:, 1, 0])

    return out


def explain(x: pd.DataFrame, target: str) -> tuple[float, np.ndarray]:
    """SHAP contributions for one building, averaged over the five fold models.

    Returns ``(base_value, values)`` in log space -- the space the model works in.
    ``base_value + values.sum()`` equals the log prediction exactly, because the
    prediction is itself the mean of the five models being explained.
    """
    import shap  # heavy import, only paid for when an explanation is requested

    models, _ = load()
    estimators = models[target]._mapie_regressor.estimator_.estimators_

    bases, contributions = [], []
    for estimator in estimators:
        explanation = shap.TreeExplainer(estimator)(x)
        bases.append(float(explanation.base_values[0]))
        contributions.append(explanation.values[0])

    return float(np.mean(bases)), np.mean(contributions, axis=0)
