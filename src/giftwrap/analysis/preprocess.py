import anndata as ad
import numpy as np


def filter_gapfills(adata: ad.AnnData,
                    min_cells: int = 10,
                    min_supporting_umis: int = 1,
                    min_supporting_reads: int = 1,
                    min_supporting_percent: float = 0.0) -> ad.AnnData:
    """
    Filter gapfills (and remove the filtered features). This can be used to remove low-quality/uncertain gapfills.
    Note that the default settings are relatively lenient.
    :param adata: The AnnData object containing the gapfills.
    :param min_cells: The minimum number of cells that a gapfill must be present in to be considered real.
    :param min_supporting_umis: The minimum number of UMIs that a gapfill must have to be considered real.
    :param min_supporting_reads: The minimum number of reads (including PCR duplicates) that a gapfill must have to be considered real.
    :param min_supporting_percent: The minimum percentage of reads that a gapfill must have to be considered real.
    :return: Returns the same AnnData object with the filtered gapfills removed.
    """
    # First set counts to 0 for all gapfills that don't meet basic criteria. Then filter by cells.
    adata.X[adata.layers["total_reads"] < min_supporting_reads] = 0
    adata.X[adata.layers["percent_supporting"] < min_supporting_percent] = 0
    adata.X[adata.X < min_supporting_umis] = 0

    # Filter by cells
    to_remove = (adata.X > 0).sum(axis=0) < min_cells
    adata = adata[:, ~to_remove]

    return adata


def filter_genotypes(adata: ad.AnnData,
                     min_umis_per_cell: int = 1,
                     min_cells: int = 1,
                     min_proportion: float = 0.0,
                     top_n: int = None
                     ) -> ad.AnnData:
    """
    Filter called genotypes by masking with NaNs. This can be used to remove low-quality/uncertain genotypes.
    Note: giftwrap.tl.call_genotypes must have been called.
    :param adata: The AnnData object containing the genotypes.
    :param min_umis_per_cell: The minimum number of UMIs per cell that a genotype must have to be considered real.
    :param min_cells: The minimum number of cells that a genotype
    :param min_proportion: The minimum proportion of UMIs that a genotype must have to be considered real.
    :param top_n: The number of top genotypes to keep. If None, all genotypes are kept.
    :return: Returns the same AnnData object with the filtered genotypes masked.
    """

    assert "genotype" in adata.obsm, "Genotypes not found in adata. Please run call_genotypes first."

    genotype_df = adata.obsm["genotype"].copy()  # Columns = probe, rows = cell, values = genotype string
    genotype_p_df = adata.obsm["genotype_proportion"].copy()
    genotype_counts_df = adata.obsm["genotype_counts"].copy()

    # Mask genotypes that don't meet basic criteria
    keep_mask = np.ones_like(genotype_df, dtype=bool)
    keep_mask[np.nan_to_num(genotype_counts_df.values) < min_umis_per_cell] = False
    keep_mask[np.nan_to_num(genotype_p_df.values) < min_proportion] = False
    # Apply
    genotype_df[~keep_mask] = np.nan
    genotype_p_df[~keep_mask] = np.nan
    genotype_counts_df[~keep_mask] = np.nan

    # Get the number of cells for each genotype
    for probe in genotype_df.columns:
        counts = genotype_df[probe].value_counts()
        # Ignore NaN
        counts = counts.dropna()
        to_remove = []
        if top_n is not None:
            # Pop the top_n
            to_remove = counts[top_n:].index.values.tolist()
            counts = counts.iloc[:top_n]

        to_remove += counts[counts < min_cells].index.values.tolist()

        # Mask
        for genotype in to_remove:
            genotype_df[genotype_df == genotype] = np.nan
            genotype_p_df[genotype_df == genotype] = np.nan
            genotype_counts_df[genotype_df == genotype] = np.nan

    # Finally re-add the masked genotypes
    adata.obsm["genotype"] = genotype_df
    adata.obsm["genotype_proportion"] = genotype_p_df
    adata.obsm["genotype_counts"] = genotype_counts_df

    return adata
