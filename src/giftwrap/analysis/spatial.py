import anndata as ad

def bin(adata: ad.AnnData, resolution: int = 8) -> ad.AnnData:
    """
    This is ONLY APPLICABLE FOR VISIUM-HD.
    This bins/aggregates data from 2 micron resolution to any other resolution (must be a power of 2).
    :param adata: The Spatial gapfill data.
    :param resolution: The resolution to aggregate into in microns. Spaceranger typically aggregates to 8um and 16 um.
    :return: The binned data.
    """
    assert resolution % 2 == 0, "Resolution must be a power of 2."
    assert resolution > 2, "Resolution must be greater than 2."