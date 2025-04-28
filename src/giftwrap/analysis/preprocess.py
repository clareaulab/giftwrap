import anndata as ad
import numpy as np


import anndata as ad
from scipy.sparse import issparse

def filter_gapfills(adata: ad.AnnData,
                    min_cells: int = 10,
                    min_supporting_umis: int = 1,
                    min_supporting_reads: int = 1,
                    min_supporting_percent: float = 0.0) -> ad.AnnData:
    """
    Filter gapfills (and remove the filtered features).
    This can be used to remove low-quality/uncertain gapfills.

    :param adata: The AnnData object containing the gapfills.
    :param min_cells: The minimum number of cells that a gapfill must be present in to be considered real.
    :param min_supporting_umis: The minimum number of UMIs that a gapfill must have to be considered real.
    :param min_supporting_reads: The minimum number of reads (including PCR duplicates)
                                 that a gapfill must have to be considered real.
    :param min_supporting_percent: The minimum percentage of reads that a gapfill must have to be considered real.
    :return: Returns the same AnnData object with the filtered gapfills removed.
    """
    # 1) Mask by total reads
    if issparse(adata.layers["total_reads"]):
        # Create a copy of the sparse matrix and replace data with booleans
        mask = adata.layers["total_reads"].copy()
        mask.data = (mask.data >= min_supporting_reads)
        adata.X = adata.X.multiply(mask)
    else:
        # If it's dense, safe to do a direct comparison
        adata.X[adata.layers["total_reads"] < min_supporting_reads] = 0

    # 2) Mask by supporting percent
    if issparse(adata.layers["percent_supporting"]):
        mask = adata.layers["percent_supporting"].copy()
        mask.data = (mask.data >= min_supporting_percent)
        adata.X = adata.X.multiply(mask)
    else:
        adata.X[adata.layers["percent_supporting"] < min_supporting_percent] = 0

    # 3) Mask by supporting UMIs (this time the comparison is on adata.X itself)
    if issparse(adata.X):
        mask = adata.X.copy()
        mask.data = (mask.data >= min_supporting_umis)
        adata.X = adata.X.multiply(mask)
    else:
        adata.X[adata.X < min_supporting_umis] = 0

    # Finally, filter by the minimum number of cells
    if issparse(adata.X):
        # Compute the number of non-zero entries in dim 1 (columns)
        counts = adata.X.getnnz(axis=0).flatten()
    else:
        # Count the number of non-zero entries in each column
        counts = np.count_nonzero(adata.X, axis=0)
    # Create a mask for columns (features) that meet the minimum cell count
    keep_mask = counts >= min_cells
    # Apply the mask to the data
    adata = adata[:, keep_mask]

    return adata



def filter_genotypes(adata: ad.AnnData,
                     min_umis_per_cell: int = 1,
                     min_cells: int = 10,
                     min_proportion: float = 0.0,
                     top_n: int = None
                     ) -> ad.AnnData:
    """
    Filter called genotypes by masking with NaNs. This can be used to remove low-quality/uncertain genotypes.
    Note: giftwrap.tl.call_genotypes must have been called.
    :param adata: The AnnData object containing the genotypes.
    :param min_umis_per_cell: The minimum number of UMIs per cell that a genotype must have to be considered real.
    :param min_cells: The minimum number of cells that a genotype must appear in to be kept.
    :param min_proportion: The minimum proportion of UMIs that a genotype must have to be considered real.
    :param top_n: The number of top genotypes to keep (by frequency). If None, keeps all that meet other criteria.
    :return: Returns the same AnnData object with the filtered genotypes masked.
    """

    assert "genotype" in adata.obsm, "Genotypes not found in adata. Please run call_genotypes first."

    genotype_df = adata.obsm["genotype"].copy()  # columns = probe, rows = cell, values = genotype string
    genotype_p_df = adata.obsm["genotype_proportion"].copy()
    genotype_counts_df = adata.obsm["genotype_counts"].copy()

    # Mask genotypes that don't meet basic per-cell criteria
    keep_mask = np.ones_like(genotype_df, dtype=bool)
    keep_mask[np.nan_to_num(genotype_counts_df.values) < min_umis_per_cell] = False
    keep_mask[np.nan_to_num(genotype_p_df.values) < min_proportion] = False

    # Apply these masks
    genotype_df[~keep_mask] = np.nan
    genotype_p_df[~keep_mask] = np.nan
    genotype_counts_df[~keep_mask] = np.nan

    # Vectorized filtering by probe:
    for probe in genotype_df.columns:
        # Count how often each genotype appears (ignoring NaN)
        counts = genotype_df[probe].value_counts(dropna=True)

        # If top_n is specified, keep only the top_n genotypes
        if top_n is not None:
            # value_counts() is already sorted in descending order by default
            counts = counts.iloc[:top_n]

        # Now also remove genotypes that do not appear in at least min_cells cells
        counts = counts[counts >= min_cells]
        keep_genotypes = counts.index  # The genotypes that pass all criteria for this probe

        # Use .isin() to vectorize the removal of failing genotypes
        mask = genotype_df[probe].isin(keep_genotypes)
        genotype_df.loc[~mask, probe] = np.nan
        genotype_p_df.loc[~mask, probe] = np.nan
        genotype_counts_df.loc[~mask, probe] = np.nan

    # Finally, store the updated data back in adata
    adata.obsm["genotype"] = genotype_df
    adata.obsm["genotype_proportion"] = genotype_p_df
    adata.obsm["genotype_counts"] = genotype_counts_df

    return adata
