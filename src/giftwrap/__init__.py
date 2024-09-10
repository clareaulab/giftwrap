# Random patches to assist with package compatibility
import numpy as np
np.float_ = np.float64
np.infty = np.inf

from .utils import read_h5_file, merge_anndatas, filter_h5_file, TechnologyFormatInfo, PrefixTree
from .analysis import preprocess as pp, plots as pl, tools as tl

__all__ = ['read_h5_file', 'merge_anndatas', 'filter_h5_file', 'TechnologyFormatInfo', "PrefixTree", 'pp', 'pl', 'tl']