# h5ad output and TCAT names

import re


def write_h5ad(adata, path, overwrite):
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Use --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path)


def safe_name(value):
    name = re.sub(r"[^0-9A-Za-z_]+", "_", str(value)).strip("_")
    return name or "unnamed"


def feature_kind(feature):
    if feature == "O_lambda_inf":
        return "spectral"
    if feature.startswith("TCAT_GEP_"):
        return "GEP"
    if "_proportion_" in feature:
        return "cell_type"
    return "score"
