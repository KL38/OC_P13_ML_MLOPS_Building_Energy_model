"""Feature engineering shared between training and inference.

This module is the single source of truth for turning building attributes into
the matrix the models consume. It is imported by the training script *and* by
the Streamlit app, so a building typed into the UI goes through exactly the same
transformations as a row of the training set.

Three entry points:

- ``from_benchmark(df)``    -- training only: turns raw Seattle benchmark rows
  into the neutral *description* both paths share (energy consumption columns
  become presence booleans, out-of-domain rows are dropped).
- ``build_features(df)``    -- batch: description DataFrame -> feature matrix.
- ``build_features_one(...)`` -- a single building described by the UI form.

Both builders funnel through ``_row_features`` so that there is one and only one
assembly path, and both return ``FEATURE_COLUMNS`` in that exact order.

Design rules:

- No fitted state. The category vocabularies below are frozen constants,
  versioned with the code.
- Unknown categories raise. A silently mis-encoded building is worse than a
  refused prediction.
- Energy consumption columns never reach the model: their sum *is* the target.

Differences with the P3 notebook are logged in ``docs/ecarts_vs_P3.md``.
"""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

# --- Vocabularies -----------------------------------------------------------

# 13 neighbourhoods, after the case normalisation done in the EDA notebook (E07).
NEIGHBOURHOODS: tuple[str, ...] = (
    "BALLARD",
    "CENTRAL",
    "DELRIDGE",
    "DOWNTOWN",
    "EAST",
    "GREATER DUWAMISH",
    "LAKE UNION",
    "MAGNOLIA / QUEEN ANNE",
    "NORTH",
    "NORTHEAST",
    "NORTHWEST",
    "SOUTHEAST",
    "SOUTHWEST",
)

# 64 Seattle use types collapsed into 11 groups. The grouping is by energy
# behaviour, not by administrative label: on 1656 rows it injects domain
# knowledge a model could not learn from the data alone.
USAGE_MAP: dict[str, str] = {
    # Offices - standard intensity, daytime hours
    "Office": "t_Office",
    "Medical Office": "t_Office",
    "Financial Office": "t_Office",
    "Bank Branch": "t_Office",
    "Courthouse": "t_Office",
    # Added by us: the E01 fix copies PrimaryPropertyType into the use-type
    # column, and that column uses a different vocabulary. P3 never hit this
    # because it wrongly copied "Hotel", which exists in both.
    "Small- and Mid-Sized Office": "t_Office",
    # Food service - very high intensity: cooking, refrigeration, ventilation
    "Restaurant": "t_Food_Service",
    "Food Service": "t_Food_Service",
    "Other - Restaurant/Bar": "t_Food_Service",
    "Bar/Nightclub": "t_Food_Service",
    "Fast Food Restaurant": "t_Food_Service",
    "Food Sales": "t_Food_Service",
    "Supermarket/Grocery Store": "t_Food_Service",  # massive food refrigeration
    # Retail and commercial - medium intensity: lighting and HVAC
    "Retail Store": "t_Commercial",
    "Strip Mall": "t_Commercial",
    "Wholesale Club/Supercenter": "t_Commercial",
    "Automobile Dealership": "t_Commercial",
    "Convenience Store without Gas Station": "t_Commercial",
    "Enclosed Mall": "t_Commercial",
    "Lifestyle Center": "t_Commercial",
    "Other - Mall": "t_Commercial",
    "Personal Services (Health/Beauty, Dry Cleaning, etc)": "t_Commercial",
    "Repair Services (Vehicle, Shoe, Locksmith, etc)": "t_Commercial",
    # Education - intermittent, idle in summer and at weekends
    "K-12 School": "t_Education",
    "College/University": "t_Education",
    "Other - Education": "t_Education",
    "Adult Education": "t_Education",
    "Vocational School": "t_Education",
    "Pre-school/Daycare": "t_Education",
    # Public safety and 24/7 services - high load factor
    "Fire Station": "t_Public_Safety",
    "Police Station": "t_Public_Safety",
    "Prison/Incarceration": "t_Public_Safety",
    "Other - Public Services": "t_Public_Safety",
    "Other - Utility": "t_Public_Safety",
    # Dry storage - very low intensity per square foot
    "Non-Refrigerated Warehouse": "t_Storage_Dry",
    "Distribution Center": "t_Storage_Dry",
    "Self-Storage Facility": "t_Storage_Dry",
    # Lodging - constant hot water, heating and cooling
    "Hotel": "t_Lodging",
    "Multifamily Housing": "t_Lodging",
    "Residence Hall/Dormitory": "t_Lodging",
    "Senior Care Community": "t_Lodging",
    "Other - Lodging/Residential": "t_Lodging",
    "Residential Care Facility": "t_Lodging",
    # High intensity - heavy industrial or medical processes
    "Data Center": "t_High_Intensity",
    "Laboratory": "t_High_Intensity",
    "Hospital (General Medical & Surgical)": "t_High_Intensity",
    "Other/Specialty Hospital": "t_High_Intensity",
    "Urgent Care/Clinic/Other Outpatient": "t_High_Intensity",
    "Other - Technology/Science": "t_High_Intensity",
    "Refrigerated Warehouse": "t_High_Intensity",
    "Manufacturing/Industrial Plant": "t_High_Intensity",
    # Assembly and recreation - large volumes to heat and light
    "Other - Recreation": "t_Assembly",
    "Other - Entertainment/Public Assembly": "t_Assembly",
    "Fitness Center/Health Club/Gym": "t_Assembly",
    "Swimming Pool": "t_Assembly",
    "Performing Arts": "t_Assembly",
    "Movie Theater": "t_Assembly",
    "Worship Facility": "t_Assembly",
    "Library": "t_Assembly",
    "Museum": "t_Assembly",
    "Social/Meeting Hall": "t_Assembly",
    # Special cases
    "Parking": "t_Parking",
    "Other": "t_Others",
    "Other - Services": "t_Others",
}

USAGE_GROUPS: tuple[str, ...] = tuple(sorted(set(USAGE_MAP.values())))

# Raw columns holding the energy consumption. Derived into presence booleans,
# then dropped: they must never reach the model.
ENERGY_COLUMNS = ("Electricity(kBtu)", "NaturalGas(kBtu)", "SteamUse(kBtu)")

# The three use-type slots published by Seattle, as (type, surface) pairs.
USAGE_SLOTS = (
    ("LargestPropertyUseType", "LargestPropertyUseTypeGFA"),
    ("SecondLargestPropertyUseType", "SecondLargestPropertyUseTypeGFA"),
    ("ThirdLargestPropertyUseType", "ThirdLargestPropertyUseTypeGFA"),
)

# Target columns, as named in dfclean.csv. P3 renamed the first one; we keep the
# source name so the cleaned file stays comparable to the raw benchmark.
TARGETS = ("SiteEnergyUse(kBtu)", "TotalGHGEmissions")


def _neighbourhood_column(name: str) -> str:
    """Column name for a neighbourhood one-hot, e.g. 'MAGNOLIA / QUEEN ANNE'."""
    slug = name.replace(" / ", "_").replace(" ", "_")
    return f"nb_{slug}"


FEATURE_COLUMNS: tuple[str, ...] = (
    "YearBuilt",
    "NumberofBuildings",
    "NumberofFloors",
    "logGFAtotal",
    "NbUsage",
    "HasElectricity",
    "HasGas",
    "HasSteam",
    *(_neighbourhood_column(n) for n in NEIGHBOURHOODS),
    *USAGE_GROUPS,
)


# --- Shared building blocks -------------------------------------------------


def usage_composition(pairs: Sequence[tuple[str, float]]) -> dict[str, float]:
    """Share of each usage group, normalised so the shares sum to 1.

    ``pairs`` are ``(use type, weight)``; weights may be surfaces or percentages,
    only their ratios matter. Normalising on the sum of the weights -- rather
    than on the building's total surface as P3 did -- keeps the vector a proper
    composition and matches what the UI collects (E08).
    """
    shares = dict.fromkeys(USAGE_GROUPS, 0.0)
    total = 0.0

    for use_type, weight in pairs:
        if use_type is None or (isinstance(weight, float) and np.isnan(weight)):
            continue
        if weight <= 0:
            continue
        try:
            group = USAGE_MAP[use_type]
        except KeyError:
            raise ValueError(
                f"Unknown property use type: {use_type!r}. "
                "Add it to USAGE_MAP with the group matching its energy profile."
            ) from None
        shares[group] += weight
        total += weight

    if total == 0:
        raise ValueError("A building must declare at least one use type with a surface")

    return {group: share / total for group, share in shares.items()}


def _row_features(desc: Mapping, composition: dict[str, float]) -> dict:
    """Assemble one feature row. Single assembly path for both entry points."""
    neighbourhood = desc["Neighborhood"]
    if neighbourhood not in NEIGHBOURHOODS:
        raise ValueError(
            f"Unknown neighbourhood: {neighbourhood!r}. Expected one of {NEIGHBOURHOODS}"
        )

    gfa_total = float(desc["PropertyGFATotal"])
    if gfa_total <= 0:
        raise ValueError("PropertyGFATotal must be strictly positive")

    row = {
        "YearBuilt": int(desc["YearBuilt"]),
        "NumberofBuildings": float(desc["NumberofBuildings"]),
        "NumberofFloors": float(desc["NumberofFloors"]),
        "logGFAtotal": float(np.log1p(gfa_total)),
        "NbUsage": int(desc["NbUsage"]),
        "HasElectricity": int(bool(desc["HasElectricity"])),
        "HasGas": int(bool(desc["HasGas"])),
        "HasSteam": int(bool(desc["HasSteam"])),
    }
    for name in NEIGHBOURHOODS:
        row[_neighbourhood_column(name)] = int(name == neighbourhood)
    row.update(composition)
    return row


# --- Entry points -----------------------------------------------------------


def from_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    """Raw Seattle benchmark rows -> the neutral description both paths share.

    Training only. Turns the energy consumption columns into presence booleans
    and drops them, then drops buildings with no usage breakdown at all: the UI
    cannot produce such a building, so it is outside the model's domain.
    """
    out = df.copy()

    out["HasElectricity"] = out["Electricity(kBtu)"] > 0
    out["HasGas"] = out["NaturalGas(kBtu)"] > 0
    out["HasSteam"] = out["SteamUse(kBtu)"] > 0
    out = out.drop(columns=list(ENERGY_COLUMNS))

    out["NbUsage"] = out["ListOfAllPropertyUseTypes"].str.count(",") + 1

    declared = out[[gfa for _, gfa in USAGE_SLOTS]].fillna(0).sum(axis=1)
    without_usage = declared <= 0
    if bool(without_usage.any()):
        dropped = list(out.loc[without_usage, "OSEBuildingID"])
        print(f"Dropped {len(dropped)} building(s) with no usage breakdown: {dropped}")
        out = out.loc[~without_usage].reset_index(drop=True)

    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Description DataFrame -> feature matrix, in FEATURE_COLUMNS order."""
    rows = []
    for record in df.to_dict("records"):
        pairs = [(record[use], record[gfa]) for use, gfa in USAGE_SLOTS]
        rows.append(_row_features(record, usage_composition(pairs)))

    return pd.DataFrame(rows, columns=list(FEATURE_COLUMNS), index=df.index)


def build_features_one(
    *,
    year_built: int,
    number_of_buildings: int,
    number_of_floors: int,
    gfa_total: float,
    neighbourhood: str,
    usages: Sequence[tuple[str, float]],
    nb_usages: int | None = None,
    has_electricity: bool = True,
    has_gas: bool = False,
    has_steam: bool = False,
) -> pd.DataFrame:
    """A single building from the UI form -> a one-row feature matrix.

    ``usages`` is a list of ``(use type, share)``; any number of entries is
    accepted, unlike the three slots Seattle publishes. ``nb_usages`` defaults to
    the number of usages declared.
    """
    desc = {
        "YearBuilt": year_built,
        "NumberofBuildings": number_of_buildings,
        "NumberofFloors": number_of_floors,
        "PropertyGFATotal": gfa_total,
        "Neighborhood": neighbourhood,
        "NbUsage": len(usages) if nb_usages is None else nb_usages,
        "HasElectricity": has_electricity,
        "HasGas": has_gas,
        "HasSteam": has_steam,
    }
    row = _row_features(desc, usage_composition(usages))
    return pd.DataFrame([row], columns=list(FEATURE_COLUMNS))
