import anndata as ad


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
