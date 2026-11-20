from pathlib import Path

import pandas as pd


ASA_COMPONENTS = [
    "TCAT_GEP_OX40_EBI3",
    "TCAT_GEP_ICOS_CD38",
    "TCAT_GEP_CTLA4_CD38",
    "TCAT_GEP_TIMD4_TIM3",
]
PROLIFERATION_COMPONENTS = [
    "TCAT_GEP_CellCycle_G2M",
    "TCAT_GEP_CellCycle_Late_S",
    "TCAT_GEP_CellCycle_S",
]

ASA_GEP_FEATURES = [feature + "_CLR" for feature in ASA_COMPONENTS]
SCORE_FEATURES = ["TCAT_SCORE_ASA_CLR", "TCAT_SCORE_Proliferation_CLR"]

labels = pd.read_csv(Path(__file__).with_name("feature_labels.csv")).set_index("feature").label.to_dict()


def plot_label(feature):
    if feature == "O_lambda":
        return r"$\mathcal{O}_{\lambda}$"
    label = labels.get(feature)
    if feature == "TCAT_SCORE_ASA_CLR":
        return label.replace(" ", "\n", 1)
    if label is not None:
        return label
    for prefix in ("TCAT_GEP_", "ATLAS_proportion_"):
        if feature.startswith(prefix):
            return feature.removeprefix(prefix).removesuffix("_CLR").replace("_", " ")
    return feature
