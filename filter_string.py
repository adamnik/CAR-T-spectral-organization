# Map gene names to the human STRING reference

import argparse
import re
import shutil
import urllib.request
from collections import defaultdict
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from analysis_io import write_h5ad


CURATED_ALIAS_SUM_GROUPS = {
    "SEPTIN4": {"SEPT4", "C17ORF47"},
    "RAMAC": {"FAM103A1", "RAMMET"},
    "MTLN": {"LINC00116", "SMIM37"},
}


def clean_text(value):
    if value is None or pd.isna(value):
        return ""
    return str(value).replace('"', "").strip()


def normalize(value):
    return clean_text(value).upper()


def normalize_without_punctuation(value):
    return re.sub(r"[-_.]", "", normalize(value))


def plausible_alias(value):
    return bool(
        value
        and len(value) <= 100
        and not any(character.isspace() for character in value)
        and ":" not in value
        and "=" not in value
        and re.fullmatch(r"[A-Za-z0-9_.-]+", value)
    )


def download_string_file(url, destination, refresh):
    if destination.exists() and destination.stat().st_size > 0 and not refresh:
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={'User-Agent': 'CAR-T-spectral-organization/1.0'})
    print(f"Downloading {url}")
    with urllib.request.urlopen(request, timeout=300) as response:
        with temporary_path.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
    temporary_path.replace(destination)


def string_paths(string_dir, version, taxon_id, refresh):
    base_url = "https://stringdb-downloads.org/download"
    info_name = f"{taxon_id}.protein.info.v{version}.txt.gz"
    aliases_name = f"{taxon_id}.protein.aliases.v{version}.txt.gz"
    info_url = f"{base_url}/protein.info.v{version}/{info_name}"
    aliases_url = f"{base_url}/protein.aliases.v{version}/{aliases_name}"

    info_path = string_dir / info_name
    aliases_path = string_dir / aliases_name
    download_string_file(info_url, info_path, refresh)
    download_string_file(aliases_url, aliases_path, refresh)
    return info_path, aliases_path


def load_string_reference(info_path, aliases_path, input_gene_names, taxon_id):
    protein_info = pd.read_csv(
        info_path, sep='\t', compression='gzip', dtype=str, keep_default_na=False
    )
    protein_info = protein_info.rename(columns={'#string_protein_id': 'string_protein_id'})

    protein_info["protein_id"] = protein_info["string_protein_id"].map(normalize)
    protein_info["preferred_original"] = protein_info["preferred_name"].map(clean_text)
    protein_info["preferred"] = protein_info["preferred_original"].map(normalize)
    protein_info = protein_info.loc[
        protein_info["protein_id"].str.startswith(f"{taxon_id}.")
        & protein_info["preferred_original"].map(plausible_alias)
    ].copy()

    if protein_info["preferred"].duplicated().any():
        duplicated = protein_info.loc[
            protein_info["preferred"].duplicated(keep=False),
            ["preferred_original", "string_protein_id"],
        ]
        raise ValueError(
            "STRING preferred names are not unique after case normalization:\n"
            + duplicated.head(20).to_string(index=False)
        )

    preferred_names = protein_info.set_index("preferred")["preferred_original"].to_dict()
    protein_to_preferred = protein_info.set_index("protein_id")["preferred"].to_dict()
    preferred_set = set(preferred_names)

    # Look up aliases only for genes without a preferred-name match
    input_tokens = {normalize(gene) for gene in input_gene_names}
    aliases_needed = input_tokens.difference(preferred_set)
    alias_candidates = defaultdict(set)

    aliases_reader = pd.read_csv(
        aliases_path,
        sep="\t",
        compression="gzip",
        dtype=str,
        keep_default_na=False,
        chunksize=500_000,
    )
    print("Reading STRING aliases needed by this dataset...")
    for chunk in aliases_reader:
        chunk = chunk.rename(columns={"#string_protein_id": "string_protein_id"})
        for protein_id, alias in chunk[['string_protein_id', 'alias']].itertuples(index=False, name=None):
            alias_token = normalize(alias)
            if alias_token not in aliases_needed or not plausible_alias(alias_token):
                continue
            canonical = protein_to_preferred.get(normalize(protein_id))
            if canonical is not None:
                alias_candidates[alias_token].add(canonical)

    alias_to_preferred = {
        alias: next(iter(candidates))
        for alias, candidates in alias_candidates.items()
        if len(candidates) == 1 and alias not in preferred_set
    }
    ambiguous = sum(len(candidates) > 1 for candidates in alias_candidates.values())

    print(f"Canonical STRING genes: {len(preferred_names):,}")
    print(f"Unambiguous aliases used by this dataset: {len(alias_to_preferred):,}")
    print(f"Ambiguous input aliases excluded: {ambiguous:,}")
    return preferred_names, preferred_set, alias_to_preferred


def columns_with_expression(matrix):
    if sp.issparse(matrix):
        matrix = matrix.tocsc(copy=True)
        matrix.eliminate_zeros()
        return np.diff(matrix.indptr) > 0
    return np.any(np.asarray(matrix) != 0, axis=0)


def columns_are_equal(matrix, first, second):
    if sp.issparse(matrix):
        difference = matrix[:, first] - matrix[:, second]
        difference.eliminate_zeros()
        return difference.nnz == 0
    first_column = np.asarray(matrix[:, first]).ravel()
    second_column = np.asarray(matrix[:, second]).ravel()
    return bool(np.array_equal(first_column, second_column))


def candidate_priority(candidate, canonical_gene):
    if candidate["match_type"] == "exact_preferred":
        priority = 0
    elif candidate["match_type"] == "casefold_preferred":
        priority = 1
    elif normalize_without_punctuation(candidate["label"]) == normalize_without_punctuation(
        canonical_gene
    ):
        priority = 2
    else:
        priority = 3
    return priority, candidate["index"]


def identical_columns(matrix, candidates):
    reference = candidates[0]["index"]
    return all(columns_are_equal(matrix, reference, candidate["index"]) for candidate in candidates[1:])


def resolve_string_columns(matrix, gene_names, preferred_names, preferred_set, aliases):
    has_expression = columns_with_expression(matrix)
    candidates_by_gene = defaultdict(list)
    unmatched = 0

    for index, label in enumerate(gene_names):
        original = clean_text(label)
        token = normalize(original)

        if token in preferred_set:
            canonical = token
            preferred = preferred_names[canonical]
            match_type = "exact_preferred" if original == preferred else "casefold_preferred"
        else:
            canonical = aliases.get(token)
            match_type = "unambiguous_alias" if canonical is not None else "unmatched"

        if canonical is None:
            unmatched += 1
            continue

        candidates_by_gene[canonical].append(
            {
                "index": index,
                "label": original,
                "match_type": match_type,
                "has_expression": bool(has_expression[index]),
            }
        )

    selected = []
    conflicts = []

    for canonical, candidates in candidates_by_gene.items():
        canonical_gene = preferred_names[canonical]
        nonzero = [candidate for candidate in candidates if candidate["has_expression"]]
        curated_aliases = CURATED_ALIAS_SUM_GROUPS.get(canonical_gene.upper())

        # Sum distinct expression columns only for these curated alias groups
        if curated_aliases is not None:
            allowed = {normalize(label) for label in curated_aliases}
            allowed.add(normalize(canonical_gene))
            curated_nonzero = []
            other_nonzero = []
            for candidate in nonzero:
                if normalize(candidate['label']) in allowed:
                    curated_nonzero.append(candidate)
                else:
                    other_nonzero.append(candidate)

            if curated_nonzero and other_nonzero:
                conflicts.append((canonical_gene, nonzero))
                continue
            if len(curated_nonzero) == 1:
                selected.append((canonical_gene, [curated_nonzero[0]["index"]]))
                continue
            if len(curated_nonzero) > 1:
                if identical_columns(matrix, curated_nonzero):
                    keep = min(
                        curated_nonzero,
                        key=lambda candidate: candidate_priority(candidate, canonical_gene),
                    )
                    selected.append((canonical_gene, [keep["index"]]))
                else:
                    selected.append(
                        (canonical_gene, sorted(candidate["index"] for candidate in curated_nonzero))
                    )
                continue

        # Prefer the canonical name over an alias
        identity = [
            candidate
            for candidate in candidates
            if candidate["match_type"] in {"exact_preferred", "casefold_preferred"}
        ]
        punctuation_equivalent = [
            candidate
            for candidate in candidates
            if candidate["match_type"] == "unambiguous_alias"
            and normalize_without_punctuation(candidate["label"])
            == normalize_without_punctuation(canonical_gene)
        ]

        if identity:
            eligible = identity
        elif len(punctuation_equivalent) == 1:
            selected.append((canonical_gene, [punctuation_equivalent[0]["index"]]))
            continue
        else:
            eligible = candidates

        nonzero_eligible = [candidate for candidate in eligible if candidate["has_expression"]]
        if len(nonzero_eligible) == 1:
            keep = nonzero_eligible[0]
        elif len(nonzero_eligible) == 0:
            keep = min(eligible, key=lambda candidate: candidate_priority(candidate, canonical_gene))
        elif identical_columns(matrix, nonzero_eligible):
            keep = min(
                nonzero_eligible,
                key=lambda candidate: candidate_priority(candidate, canonical_gene),
            )
        else:
            conflicts.append((canonical_gene, nonzero_eligible))
            continue

        selected.append((canonical_gene, [keep["index"]]))

    if conflicts:
        examples = []
        for canonical_gene, candidates in conflicts[:20]:
            labels = ", ".join(candidate["label"] for candidate in candidates)
            examples.append(f"  {canonical_gene}: {labels}")
        raise ValueError(
            "Conflicting nonidentical features map to the same STRING gene. "
            "Resolve these inputs explicitly before continuing:\n" + "\n".join(examples)
        )

    selected.sort(key=lambda item: item[0])
    print(f"Unmatched input genes removed: {unmatched:,}")
    print(f"Canonical STRING genes retained: {len(selected):,}")
    return selected


def build_filtered_matrix(matrix, selected):
    if sp.issparse(matrix):
        source_indices = []
        target_indices = []
        for target, (_, indices) in enumerate(selected):
            source_indices.extend(indices)
            target_indices.extend([target] * len(indices))

        projection = sp.csr_matrix(
            (np.ones(len(source_indices), dtype=matrix.dtype), (source_indices, target_indices)),
            shape=(matrix.shape[1], len(selected)),
        )
        filtered = (matrix @ projection).tocsr()
        filtered.sum_duplicates()
        filtered.eliminate_zeros()
        return filtered

    source = np.asarray(matrix)
    filtered = np.zeros((source.shape[0], len(selected)), dtype=source.dtype)
    for target, (_, indices) in enumerate(selected):
        for source_index in indices:
            filtered[:, target] += source[:, source_index]
    return filtered


def main():
    parser = argparse.ArgumentParser(description='Map one h5ad to unique canonical human STRING genes.')
    parser.add_argument("input_h5ad", type=Path)
    parser.add_argument("output_h5ad", type=Path)
    parser.add_argument("--string-version", default="12.0")
    parser.add_argument("--taxon-id", default="9606")
    parser.add_argument("--string-dir", type=Path, default=Path("STRING"))
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    adata = ad.read_h5ad(args.input_h5ad)
    print(f"Loaded {args.input_h5ad}: {adata.n_obs:,} cells x {adata.n_vars:,} genes")
    if not adata.obs_names.is_unique:
        adata.obs_names_make_unique()
        print("Made duplicated observation names unique.")
    if not adata.var_names.is_unique:
        raise ValueError(
            "Input gene names are duplicated. Exact duplicates must be collapsed by "
            "preprocess_h5ad.py rather than renamed with var_names_make_unique()."
        )

    info_path, aliases_path = string_paths(
        args.string_dir, args.string_version, args.taxon_id, args.refresh
    )
    preferred_names, preferred_set, aliases = load_string_reference(
        info_path,
        aliases_path,
        adata.var_names.astype(str),
        args.taxon_id,
    )
    selected = resolve_string_columns(
        adata.X, adata.var_names.astype(str), preferred_names, preferred_set, aliases
    )
    if not selected:
        raise ValueError("No genes could be mapped to STRING.")

    filtered_X = build_filtered_matrix(adata.X, selected)
    canonical_names = [gene for gene, _ in selected]
    filtered = ad.AnnData(
        X=filtered_X,
        obs=adata.obs.copy(),
        var=pd.DataFrame(index=pd.Index(canonical_names, name="gene_symbol")),
    )
    filtered.uns["STRING"] = {'version': args.string_version, 'taxon_id': args.taxon_id}

    write_h5ad(filtered, args.output_h5ad, args.overwrite)
    print(f"Saved {args.output_h5ad}: {filtered.n_obs:,} cells x {filtered.n_vars:,} genes")


if __name__ == "__main__":
    main()
