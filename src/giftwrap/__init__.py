"""
GIFTwrap: A Python package for analyzing GIFT-seq data.

The package provides both a CLI for transforming FASTQ files to counts matrices,
as well as a Python API for analysis.
"""

# Random patches to assist with package compatibility
import numpy as np
np.float_ = np.float64
np.infty = np.inf

# Silence various warnings:
try:
    import warnings

    warnings.filterwarnings(
        "ignore",
        message="Importing read_text from `anndata` is deprecated. Import anndata.io.read_text instead.",
        category=FutureWarning,
        module="anndata.utils"
    )

    warnings.filterwarnings(
        "ignore",
        message="pkg_resources is deprecated as an API",
        category=UserWarning,
    )
except: pass

try:
    import dask  # Dask is imported with the spatial extra
    dask.config.set({"dataframe.query-planning": True})
except: pass

try:
    import importlib_metadata
except ImportError:
    try:
        import importlib.metadata as importlib_metadata
    except ImportError:
        importlib_metadata = None

if importlib_metadata is not None:
    __version__ = importlib_metadata.version("giftwrap-sc")
else:
    __version__ = "unknown"

from .utils import read_h5_file, filter_h5_file_by_barcodes, TechnologyFormatInfo, sequence_saturation_curve, sequencing_saturation
from .analysis import preprocess as pp, plots as pl, tools as tl, spatial as sp


__all__ = ['read_h5_file',
           'filter_h5_file_by_barcodes',
           'TechnologyFormatInfo',
           'sequence_saturation_curve',
           'sequencing_saturation',
           'pp', 'pl', 'tl', 'sp']
