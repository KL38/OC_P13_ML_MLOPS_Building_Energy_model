"""Feature engineering shared between training and inference.

This module is the single source of truth for turning building attributes into
the matrix the models consume. It is imported by the training script *and* by
the Streamlit app, so that a building typed into the UI goes through exactly the
same transformations as a row of the training set.

Two entry points are planned:

- ``build_features(df)``   -- batch, from a ``dfclean.csv``-shaped DataFrame
- ``build_features_one(**kwargs)`` -- a single building described by the UI form

Both must return the same columns, in the same order.

Design constraints:

- No fitted state hidden in the module. Anything learned from the training set
  (category lists, column order) is either a hard-coded constant here or is
  persisted alongside the model.
- Unknown categories must fail loudly, not silently produce a wrong row.
- The reference year for the building age is the data year (2016), not the
  current year: the models were trained against that reference.

Content is being decided -- see the F1..F10 list in the session notes.
"""

REFERENCE_YEAR = 2016

TARGETS = ("SiteEnergyUse_kBtu", "TotalGHGEmissions")


def build_features(df):
    """Turn a cleaned building DataFrame into the model feature matrix."""
    raise NotImplementedError("Feature list under discussion (F1..F10)")
