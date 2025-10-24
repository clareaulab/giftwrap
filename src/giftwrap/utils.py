import tempfile
import warnings, os
from collections import defaultdict
from enum import Enum

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning")  # inherit to subprocesses

import functools
import gzip
import io
import json
import shutil
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from tempfile import NamedTemporaryFile
import itertools
import contextlib
from typing import Literal, Optional, Union, Iterable
import multiprocessing

import math
import numpy as np
import pandas as pd
import h5py
import anndata as ad
import scipy
from Bio.SeqIO.QualityIO import FastqGeneralIterator
from prefixtrie import PrefixTrie, load_shared_trie, create_shared_trie


class ReadProcessState(Enum):
    FILTERED_NO_CONSTANT = 0
    FILTERED_NO_RHS = 1
    FILTERED_NO_LHS = 2
    FILTERED_NO_PROBE_BARCODE = 3
    FILTERED_NO_CELL_BARCODE = 4
    FILTERED_INCORRECT_PROBE_BARCODE = 5
    CORRECTED_RHS = 6
    CORRECTED_LHS = 7
    CORRECTED_BARCODE = 8
    EXACT = 9
    TOTAL_READS = 10  # Placeholder so that we can count the total number of reads


#Based on: https://docs.python.org/3/library/itertools.html#itertools.batched
def batched(iterator, n):
    """
    Returns a generator that yields lists of n elements from the input iterable.
    The final list may have fewer than n elements.
    """
    while True:
        chunk = list(itertools.islice(iterator, n))
        if not chunk:
            return
        yield chunk


class DummyResult:

    def __init__(self, res):
        self.res = res

    def get(self, *args, **kwargs):
        return self.res

    def wait(self, *args, **kwargs):
        pass

    def ready(self, *args, **kwargs):
        return True

    def successful(self, *args, **kwargs):
        return True


# Inject starmap_async
class ItertoolsWrapper:

    def starmap(self, *args, **kwargs):
        return itertools.starmap(*args, **kwargs)

    def starmap_async(self, *args, **kwargs):
        return DummyResult(itertools.starmap(*args, **kwargs))


def maybe_multiprocess(cores: int) -> multiprocessing.Pool:
    """
    Return a context manager that will either return the multiprocessing module or a dummy module depending on if there
    are more than 1 core reqeusted.
    :param cores: The number of cores.
    :return: The multiprocessing module or a dummy module.
    """
    if cores > 1:
        mp = multiprocessing.Pool(cores)
    else:
        mp = contextlib.nullcontext(ItertoolsWrapper())  # No multiprocessing
    return mp


def read_manifest(output_dir: Path) -> pd.DataFrame:
    """
    Read the manifest file. This is a TSV file with the following columns:
    - index: The index for the probe
    - name: The name of the probe
    - lhs_probe: The left hand side probe sequence
    - rhs_probe: The right hand side probe sequence
    - gap_probe_sequence: The sequence the probe was designed against
    - original_gap_probe_sequence: The expected WT gap probe sequence
    :param output_dir: The pipeline output directory.
    :return: The parsed dataframe which should be indexed by the index.
    """
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
    return pd.read_table(output_dir / "manifest.tsv")


class TechnologyFormatInfo(ABC):
    """
    Generic class to hold metadata related to parsing Read1 and Read2.
    """

    def __init__(self,
                 barcode_dir: Optional[str] = None,
                 read1_length: Optional[int] = None,
                 read2_length: Optional[int] = None):
        self._read1_length = read1_length
        self._read2_length = read2_length

        if barcode_dir:
            self._barcode_dir = Path(barcode_dir)
        else:
            # Fallback to our resources directory
            self._barcode_dir = Path(__file__).parent / "resources"


    @property
    def read1_length(self) -> Optional[int]:
        """
        This is the expected length of each R1 read, if defined the pipeline can improve performance.
        :return: The length, or None if not defined.
        """
        return self._read1_length

    @property
    def read2_length(self) -> Optional[int]:
        """
        This is the expected length of each R2 read, if defined the pipeline can improve performance.
        :return: The length, or None if not defined.
        """
        return self._read2_length

    @property
    @abstractmethod
    def umi_start(self) -> int:
        """
        The start position of the UMI sequence in R1.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def umi_length(self) -> int:
        """
        The length of the UMI sequence on R1.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def cell_barcodes(self) -> list[str]:
        """
        The list of potential barcodes.
        """
        raise NotImplementedError()

    @property
    def n_barcodes(self) -> int:
        return len(self.barcode_tree)

    @property
    @abstractmethod
    def cell_barcode_start(self) -> int:
        """
        The start position of the cell barcode in the read.
        """
        raise NotImplementedError()

    @property
    @functools.lru_cache(1)
    def max_cell_barcode_length(self) -> int:
        """
        Returns the maximum length of a cell barcode.
        """
        return max(map(len, self.cell_barcodes))

    @functools.lru_cache(maxsize=1000)
    def barcode2coordinates(self, barcode: str) -> tuple[int, int]:
        """
        Returns the X and Y coordinates of a barcode.
        :param barcode: The barcode.
        """
        return self.barcode_coordinates[barcode]

    @property
    @abstractmethod
    def is_spatial(self) -> bool:
        """
        Whether the technology is spatial. If true, then barcode_coordinates() must be defined.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def barcode_coordinates(self) -> dict[str, tuple[int, int]]:
        """
        The x and y coordinates of the barcode in the read.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def constant_sequence(self) -> str:
        """
        The constant sequence that is expected in the read.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def constant_sequence_start(self) -> int:
        """
        The start position of the constant sequence in the read. Note that this should be relative to the end of the read
            insert. For example, in 10X flex, 0 would be the first base after the LHS + gapfill + RHS.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def has_constant_sequence(self) -> bool:
        """
        Whether the read has a constant sequence.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def probe_barcodes(self) -> list[str]:
        """
        The list of potential probe barcodes.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def probe_barcode_start(self) -> int:
        """
        The start position of the probe barcode in the read. Note that this should be relative to the end of the constant
            sequence insert. For example, in 10X flex, 2 would be the first base after the constant sequence+NN.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def probe_barcode_length(self) -> int:
        """
        The length of the probe barcode.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def has_probe_barcode(self) -> bool:
        """
        Whether the read has a probe barcode.
        """
        raise NotImplementedError()

    @abstractmethod
    def probe_barcode_index(self, bc: str) -> int:
        """
        Convert a probe barcode to an index.
        """
        raise NotImplementedError()

    def make_barcode_string(self, cell_barcode: str, plex: int = 1, x_coord: Optional[int] = None, y_coord: Optional[int] = None, is_multiplexed: bool = False) -> str:
        """
        Format a cell barcode into a string.
        :param cell_barcode: The barcode.
        :param plex: The bc index for representing demultiplexed cells.
        :param x_coord: The x coordinate.
        :param y_coord: The y coordinate.
        :param is_multiplexed: Whether the data is multiplexed.
        """
        return f"{cell_barcode}-{plex}"  # Naive multiplexed barcode

    @property
    @functools.lru_cache(maxsize=1)
    def barcode_tree(self) -> PrefixTrie:
        """
        Return a prefix tree (trie) of the cell barcodes for fast mismatch searches.
        :return: The tree.
        """
        return PrefixTrie(self.cell_barcodes)

    @functools.lru_cache(1024)
    def correct_barcode(self, read: str, max_mismatches: int, start_idx: int, end_idx: int) -> tuple[Optional[str], int]:
        """
        Given a probable barcode string, attempt to correct the sequence.
        :param read: The barcode-containing sequence.
        :param max_mismatches: The maximum number of mismatches to allow.
        :param start_idx: The start index of the barcode in the read.
        :param end_idx: The end index of the barcode in the read.
        :return: The corrected barcode, or None if no match was found and the number of corrections required.
        """
        return self.barcode_tree.search(read[start_idx:end_idx], max_mismatches)


_tx_barcode_oligos = {s: (i+1) for i, s in enumerate([
    "ACTTTAGG",
    "AACGGGAA",
    "AGTAGGCT",
    "ATGTTGAC",
    "ACAGACCT",
    "ATCCCAAC",
    "AAGTAGAG",
    "AGCTGTGA",
    "ACAGTCTG",
    "AGTGAGTG",
    "AGAGGCAA",
    "ACTACTCA",
    "ATACGTCA",
    "ATCATGTG",
    "AACGCCGA",
    "ATTCGGTT"
])}
_tx_barcode_to_oligo = {v: k for k, v in _tx_barcode_oligos.items()}


def _parse_possible_barcodes(barcode_lists: list[Path]) -> np.ndarray[str]:
    """
    Parse a list of barcode files into a single array.
    :param barcode_lists: The paths to read barcodes from.
    :return: A numpy array of barcodes. Or None if no barcodes were found.
    """
    barcodes = None
    for barcode_path in barcode_lists:
        try:
            to_add = read_wta(
                barcode_path,
                barcodes_only=True
            )
            # Convert the numpy array to a pandas series
            to_add = pd.Series(
                to_add
            )
            if barcodes is None:
                barcodes = to_add
            else:
                barcodes = pd.concat([barcodes, to_add], ignore_index=True)
        except:
            print(
                "Warning: Unable to parse barcodes from the provided WTA cellranger output.", barcode_path,
                " Falling back to bundled barcodes."
            )

    if barcodes is None:
        # If no barcodes were found, return None
        return None
    return barcodes.drop_duplicates().reset_index(drop=True)


class FlexFormatInfo(TechnologyFormatInfo):
    """
    Describes the format of a 10X Flex run.
    """

    def __init__(self,
                 barcode_dir: Optional[str] = None,
                 read1_length: Optional[int] = 28,
                 read2_length: Optional[int] = 90,
                 barcode_list: Optional[list[Path]] = None):
        if barcode_dir is None and barcode_list is None:
            raise ValueError("Either barcode_dir or barcode_list must be provided.")

        super().__init__(barcode_dir, read1_length, read2_length)
        if barcode_list:
            barcodes = _parse_possible_barcodes(barcode_list)
            if barcodes is not None:
                barcodes = barcodes.str[:16]  # Strip potential probe barcodes that are appended when multiplexed
        else:
            barcodes = None
        if barcodes is None:
            # Load the barcodes
            barcodes = pd.read_table(self._barcode_dir / "737K-fixed-rna-profiling.txt.gz", header=None, names=["barcode"], compression="gzip")["barcode"]

        # Strip the -Number from the barcode
        barcodes = barcodes.str.split("-").str[0]
        # Collect the universe of barcodes
        # self._barcodes = PrefixTrie(list(barcodes.unique()))
        # Shared memory prefix trie
        self._barcodes, self._trie_name = create_shared_trie(list(barcodes.unique()))
        self._probe_barcodes = _tx_barcode_oligos

        self._index_to_probe_barcodes = _tx_barcode_to_oligo

    # Custom pickling for multiprocessing
    def __getstate__(self):
        state = dict()
        # Remove the barcode tree from the state
        state['_trie_name'] = self._trie_name
        state['_barcodes'] = None
        state['_read1_length'] = self._read1_length
        state['_read2_length'] = self._read2_length
        state['_barcode_dir'] = self._barcode_dir
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Recreate the barcode tree from shared memory
        self._barcodes = load_shared_trie(self._trie_name)

    # Clean up shared memory on deletion
    def __del__(self):
        try:
            self._barcodes.cleanup_shared_memory()
        except: pass
        self._barcodes = None

    @property
    def umi_start(self) -> int:
        return 16

    @property
    def umi_length(self) -> int:
        return 12

    @property
    def cell_barcodes(self) -> list[str]:
        return self._barcodes.values()

    @property
    def cell_barcode_start(self) -> int:
        return 0

    @property
    def is_spatial(self) -> bool:
        return False

    @property
    def constant_sequence(self) -> str:
        return "ACGCGGTTAGCACGTA"

    @property
    def constant_sequence_start(self) -> int:
        return 0

    @property
    def has_constant_sequence(self) -> bool:
        return True

    @property
    @functools.lru_cache(1)
    def probe_barcodes(self) -> list[str]:
        return list(self._probe_barcodes.keys())

    @property
    def probe_barcode_start(self) -> int:
        return 2  # There is an NN between the constant sequence and the probe barcode

    @property
    def probe_barcode_length(self) -> int:
        return 8

    @property
    def has_probe_barcode(self) -> bool:
        return True

    def probe_barcode_index(self, bc: str):
        return self._probe_barcodes[bc]

    def make_barcode_string(self, cell_barcode: str, plex: int = 1, x_coord: Optional[int] = None, y_coord: Optional[int] = None, is_multiplexed: bool = False) -> str:
        if is_multiplexed:
            cell_barcode += self._index_to_probe_barcodes[plex]
        return f"{cell_barcode}-1"

    @property
    def barcode_coordinates(self) -> dict[str, tuple[int, int]]:
        raise NotImplementedError()

    @property
    def barcode_tree(self) -> PrefixTrie:
        return self._barcodes


class VisiumFormatInfo(TechnologyFormatInfo):

    def __init__(self,
                 version: int = 5,
                 barcode_dir: Optional[str] = None,
                 read1_length: Optional[int] = 28,
                 read2_length: Optional[int] = 90,
                 barcode_list: Optional[list[Path]] = None
                 ):
        if barcode_dir is None and barcode_list is None:
            raise ValueError("Either barcode_dir or barcode_list must be provided.")

        super().__init__(barcode_dir, read1_length, read2_length)
        # Load the barcodes
        if barcode_list:
            barcodes = _parse_possible_barcodes(barcode_list)
        else:
            barcodes = None
        if barcodes is None:
            # TODO: I am assuming that X is first and Y is second
            barcodes = pd.read_table(self._barcode_dir / f"visium-v{version}_coordinates.txt", header=None, names=["barcode", 'x', 'y'])
        # self._barcodes = PrefixTrie(barcodes["barcode"].tolist())
        # Shared memory prefix trie
        self._barcodes, self._trie_name = create_shared_trie(barcodes["barcode"].tolist())
        self._barcode_coordinates = {row["barcode"]: (row["x"], row["y"]) for _, row in barcodes.iterrows()}
        self._version = version

    # Custom pickling for multiprocessing
    def __getstate__(self):
        state = dict()
        # Remove the barcode tree from the state
        state['_trie_name'] = self._trie_name
        state['_barcodes'] = None
        state['_read1_length'] = self._read1_length
        state['_read2_length'] = self._read2_length
        state['_barcode_dir'] = self._barcode_dir
        state['_version'] = self._version
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Recreate the barcode tree from shared memory
        self._barcodes = load_shared_trie(self._trie_name)

    # Clean up shared memory on deletion
    def __del__(self):
        try:
            self._barcodes.cleanup_shared_memory()
        except: pass
        self._barcodes = None

    @property
    def umi_start(self) -> int:
        return 0

    @property
    def umi_length(self) -> int:
        return 12

    @property
    def cell_barcodes(self) -> list[str]:
        return self._barcodes.values()

    @property
    def cell_barcode_start(self) -> int:
        return 12

    @property
    def is_spatial(self) -> bool:
        return True

    @property
    def barcode_coordinates(self) -> dict[str, tuple[int, int]]:
        return self._barcode_coordinates

    @property
    def has_constant_sequence(self) -> bool:
        return False

    @property
    def has_probe_barcode(self) -> bool:
        return False

    @property
    def constant_sequence(self) -> str:
        raise NotImplementedError()

    @property
    def constant_sequence_start(self) -> int:
        raise NotImplementedError()

    @property
    def probe_barcodes(self) -> list[str]:
        raise NotImplementedError()

    @property
    def probe_barcode_start(self) -> int:
        raise NotImplementedError()

    def probe_barcode_index(self, bc: str):
        raise NotImplementedError()

    @property
    def probe_barcode_length(self) -> int:
        raise NotImplementedError()

    @property
    def barcode_tree(self) -> PrefixTrie:
        return self._barcodes


class VisiumHDFormatInfo(TechnologyFormatInfo):

    def __init__(self,
                 space_ranger_path: Optional[str] = None,
                 barcode_dir: Optional[str] = None,
                 read1_length: Optional[int] = 43,
                 read2_length: Optional[int] = 50,
                 barcode_list: Optional[list[Path]] = None):
        super().__init__(barcode_dir, read1_length, read2_length)

        # xy_whitelist = None
        # if barcode_list is not None:
        #     barcodes = _parse_possible_barcodes(barcode_list)
        #     if barcodes is not None:
        #         xy_whitelist = set()
        #         # Parse the coordinates from the barcode strings
        #         for bc in barcodes:
        #             y, x = bc.split("-")[0].split("_")[-2:]
        #             x = int(x)
        #             y = int(y)
        #             xy_whitelist.add((x, y))

        # Load the barcodes, note that this REQUIRES spaceranger to be installed
        import shutil
        import importlib
        import sys
        # Find spaceranger
        if not space_ranger_path:
            spaceranger = shutil.which("spaceranger")
        else:
            spaceranger = space_ranger_path
        if not spaceranger or not os.path.exists(spaceranger):
            raise FileNotFoundError("spaceranger not found on PATH.")
        spaceranger_path = Path(spaceranger)
        paths_to_add = [
            spaceranger_path.parent.parent / "lib" / "python" / "cellranger" / "spatial", # If we found the true binary
            spaceranger_path.parent / "lib" / "python" / "cellranger" / "spatial"
        ] # But spaceranger is also symlinked
        path_to_add = [p for p in paths_to_add if p.exists()]
        if len(path_to_add) == 0:
            raise FileNotFoundError("Incorrect spaceranger found on PATH.")

        # Import the protobuf schema
        sys.path.extend([str(p) for p in path_to_add])
        schema_def = importlib.import_module("visium_hd_schema_pb2")
        # Parse the schema
        slide_def = schema_def.VisiumHdSlideDesign()
        with open(self._barcode_dir / "visium_hd_v1.slide", 'rb') as f:
            slide_def.ParseFromString(f.read())

        chem_defs = Path(str(spaceranger_path.parent.parent / "lib" / "python" / "cellranger" / "chemistry_defs.json"))
        if not chem_defs.exists():
            chem_defs = self._barcode_dir / "chemistry_defs.json"

        chem_defs = json.loads(chem_defs.read_text())
        hd_def = chem_defs["SPATIAL-HD-v1"]

        segment1 = hd_def["barcode"][0]
        self._segment1_length = segment1['length']  # 14
        self._segment1_offset = segment1["offset"]  # 11
        segment2 = hd_def["barcode"][1]
        self._segment2_length = segment2['length']  # 14
        self._segment2_offset = segment2["offset"]  # 25

        extraction_params = hd_def['barcode_extraction']['params']
        self._max_offset = extraction_params['max_offset']  # 12
        self._min_offset = extraction_params['min_offset']  # 8

        # Assemble all possible barcodes
        self._barcode_coordinates = dict()
        self._bc_lengths = set()
        bc1s = set()
        bc2s = set()
        # _barcode_tree = PrefixTree([], allow_indels=True)  # Allow for indels as per 10X:
        # https://www.10xgenomics.com/support/software/space-ranger/latest/algorithms-overview/gene-expression#:~:text=using%20the%20edit%20distance%2C%20which%20allows%20for%20insertions%2C%20deletions%2C%20and%20substitutions.%20Up%20to%20four%20edits%20are%20permissible%20to%20correct%20a%20barcode%20to%20the%20whitelist.
        for x, bc1 in enumerate(slide_def.two_part.bc1_oligos):
            for y, bc2 in enumerate(slide_def.two_part.bc2_oligos):
                # if xy_whitelist is not None:
                #     if (x, y) not in xy_whitelist:
                #         continue   # Skip if not in the whitelist
                bc1s.add(bc1)  # Add barcodes if coordinate passed whitelist
                bc2s.add(bc2)
                self._bc_lengths.add(len(bc1))
                self._bc_lengths.add(len(bc2))
                cell_bc = bc1 + bc2
                self._barcode_coordinates[cell_bc] = (x, y)
                # _barcode_tree.add(cell_bc)
        self._bc_lengths = list(sorted(self._bc_lengths))
        # self._barcode_tree = SequentialPrefixTree([bc1_tree, bc2_tree])
        # self._bc1_tree = PrefixTrie(bc1s, allow_indels=True)
        # self._bc2_tree = PrefixTrie(bc2s, allow_indels=True)
        self._bc1_tree, self._bc1_tree_name = create_shared_trie(list(bc1s), allow_indels=True)
        self._bc2_tree, self._bc2_tree_name = create_shared_trie(list(bc2s), allow_indels=True)

    # Custom pickling for multiprocessing
    def __getstate__(self):
        state = dict()
        # Remove the barcode tree from the state
        state['_bc1_tree_name'] = self._bc1_tree_name
        state['_bc2_tree_name'] = self._bc2_tree_name
        state['_bc1_tree'] = None
        state['_bc2_tree'] = None
        state['_read1_length'] = self._read1_length
        state['_read2_length'] = self._read2_length
        state['_barcode_dir'] = self._barcode_dir
        state['_barcode_coordinates'] = self._barcode_coordinates
        state['_bc_lengths'] = self._bc_lengths
        state['_max_offset'] = self._max_offset
        state['_min_offset'] = self._min_offset
        state['_segment1_length'] = self._segment1_length
        state['_segment1_offset'] = self._segment1_offset
        state['_segment2_length'] = self._segment2_length
        state['_segment2_offset'] = self._segment2_offset
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Recreate the barcode trees from shared memory
        self._bc1_tree = load_shared_trie(self._bc1_tree_name)
        self._bc2_tree = load_shared_trie(self._bc2_tree_name)

    # Clean up shared memory on deletion
    def __del__(self):
        try:
            self._bc1_tree.cleanup_shared_memory()
        except: pass
        try:
            self._bc2_tree.cleanup_shared_memory()
        except: pass
        self._bc1_tree = None
        self._bc2_tree = None

    @property
    def umi_start(self) -> int:
        return 0

    @property
    def umi_length(self) -> int:
        return 9

    @property
    def cell_barcodes(self) -> list[str]:
        return list(self._barcode_coordinates.keys())

    @property
    def cell_barcode_start(self) -> int:
        return 9

    @property
    def is_spatial(self) -> bool:
        return True

    @property
    def barcode_coordinates(self) -> dict[str, tuple[int, int]]:
        return self._barcode_coordinates

    @property
    def has_constant_sequence(self) -> bool:
        return False

    @property
    def has_probe_barcode(self) -> bool:
        return False

    # Cell barcodes will be the 2um "binned" output
    def make_barcode_string(self, cell_barcode: str, plex: int = 1, x_coord: Optional[int] = None, y_coord: Optional[int] = None, is_multiplexed: bool = False) -> str:
        return f"s_002um_{y_coord:05d}_{x_coord:05d}-{plex}"

    @property
    def constant_sequence(self) -> str:
        raise NotImplementedError()

    @property
    def constant_sequence_start(self) -> int:
        raise NotImplementedError()

    @property
    def probe_barcodes(self) -> list[str]:
        raise NotImplementedError()

    @property
    def probe_barcode_start(self) -> int:
        raise NotImplementedError()

    def probe_barcode_index(self, bc: str):
        raise NotImplementedError()

    @property
    def probe_barcode_length(self) -> int:
        raise NotImplementedError()

    @property
    def n_barcodes(self) -> int:
        return len(self._barcode_coordinates)

    @property
    @functools.lru_cache(1)
    def max_cell_barcode_length(self) -> int:
        return max(self._bc_lengths) + 4  # Max number of insertions allowed

    @property
    @functools.lru_cache(1)
    def min_cell_barcode_length(self) -> int:
        return min(self._bc_lengths)

    @property
    def n_cell_barcodes(self) -> int:
        return len(self._barcode_coordinates)

    @property
    def barcode_tree(self) -> PrefixTrie:
        # NOTE: VisiumHD is weird, with hierarchical barcode lengths
        # return SequentialPrefixTree([self._bc1_tree, self._bc2_tree])
        raise NotImplementedError()

    @functools.lru_cache(1)
    def get_lengths_to_search(self):
        search_lengths = []
        for length in self._bc_lengths:
            adjusted_less = length - 1
            adjusted_more = length + 1
            if length not in search_lengths:
                search_lengths.append(length)
            if adjusted_less not in search_lengths and adjusted_less >= 0:
                search_lengths.append(adjusted_less)
            if adjusted_more not in search_lengths:
                search_lengths.append(adjusted_more)
        search_lengths.sort(reverse=True)
        return search_lengths

    @property
    @functools.lru_cache(1)
    def _default_range(self):
        return ((self._segment1_offset, self._segment1_offset + self._segment1_length),
         (self._segment2_offset, self._segment2_offset + self._segment2_length))

    @functools.lru_cache(2)
    def possible_pairing_ranges(self, read_len: int, padding: int = 0) -> list[tuple[tuple[int, int],tuple[int, int]]]:
        possible_pairing_ranges = set()
        for offset in range(self._min_offset, self._max_offset+1):
            # For each offset, try every possible combination of the two barcodes
            # for bc1_len, bc2_len in itertools.product(self._bc_lengths, self._bc_lengths):
            for bc1_len in range(min(self._bc_lengths) - padding, max(self._bc_lengths)+1 + padding):  # padded range from min length to max length+1
                for bc2_len in range(min(self._bc_lengths) - padding, max(self._bc_lengths)+1 + padding):  # padded range from min length to max length+1
                    # Skip if the length would go off the read
                    if bc1_len + bc2_len + offset > read_len:
                        continue

                    possible_pairing_ranges.add(((offset, offset+bc1_len), (offset+bc1_len, offset+bc1_len+bc2_len)))

        # Add the default range from the chemistry def. We fall back to this if there are no best matches
        possible_pairing_ranges.add(self._default_range)

        # Sort by:  total length (longest first), then by bc1 length (longest first), then by bc2 length (longest first), start position (lowest first)
        possible_pairing_ranges = list(sorted(possible_pairing_ranges, key=lambda x: (x[1][1] - x[0][0], x[0][1]-x[0][0], x[1][1]-x[1][0], -x[0][0]), reverse=True))

        return possible_pairing_ranges

    # Override the search to search all trees
    @functools.lru_cache(250_000)
    def correct_barcode(self, read: str, max_mismatches: int, start_idx: int, end_idx: int) -> tuple[Optional[str], int]:
        barcode, corrections = self._correct_barcode(read, max_mismatches, start_idx, end_idx)
        if barcode is not None:
            # Check that the barcode is in our whitelist
            if barcode not in self._barcode_coordinates:
                return None, False
        return barcode, corrections

    ## Cache bc1 searches separately
    @functools.lru_cache(125_000)
    def _cached_bc1_search(self, possible_bc1: str, max_mismatches: int) -> tuple[Optional[str], int]:
        return self._bc1_tree.search(possible_bc1, max_mismatches)

    ## Cache bc2 searches separately
    @functools.lru_cache(125_000)
    def _cached_bc2_search(self, possible_bc2: str, max_mismatches: int) -> tuple[Optional[str], int]:
        return self._bc2_tree.search(possible_bc2, max_mismatches)

    def _correct_barcode(self, read: str, max_mismatches: int, start_idx: int, end_idx: int) -> tuple[Optional[str], int]:
        # Based on logic from spaceranger: https://github.com/10XGenomics/spaceranger/blob/main/lib/rust/cr_lib/src/stages/barcode_correction.rs#L100-L235
        # and https://github.com/10XGenomics/spaceranger/blob/main/lib/rust/cr_types/src/rna_read.rs#L297-L349
        # Note that we ignore the provided start and end indices, since we will follow 10X's logic exactly

        # Additionally note that to match spaceranger we will allow for 4 mismatches
        max_mismatches = max(max_mismatches, 4)

        initial_search = self.possible_pairing_ranges(len(read), padding=0)
        # Exact search first
        valid1 = False
        valid2 = False
        bc1_range = None
        bc2_range = None
        for (possible_bc1_start, possible_bc1_end), (possible_bc2_start, possible_bc2_end) in initial_search:
            bc1, bc1_corrections = self._cached_bc1_search(read[possible_bc1_start:possible_bc1_end], 0)
            bc2, bc2_corrections = self._cached_bc2_search(read[possible_bc2_start:possible_bc2_end], 0)
            bc1_found = bc1 is not None
            bc2_found = bc2 is not None
            match (bc1_found, bc2_found):
                case (True, True):
                    return bc1 + bc2, 0  # Perfect match, return immediately
                case (True, False):
                    valid1 = True
                    bc1_range = (possible_bc1_start, possible_bc1_end)
                    valid2 = False
                    bc2_range = (possible_bc2_start, possible_bc2_end)
                    continue
                case (False, True):
                    if not valid1:  # Prefer to keep valid1 if it exists
                        valid1 = False
                        bc1_range = (possible_bc1_start, possible_bc1_end)
                        valid2 = True
                        bc2_range = (possible_bc2_start, possible_bc2_end)
                    continue
                case (False, False):
                    continue

        match (valid1, valid2):
            case (True, True):
                return bc1 + bc2, 0  # Both valid, but no perfect match, return immediately
            case (True, False):  # Only bc1 valid, search for bc2 with mismatches
                bc1_end = bc1_range[1]
                bc1 = read[bc1_range[0]:bc1_range[1]]
                # Search for bc2 with mismatches
                best_len = -1
                best_corrections = 1e6
                best_bc2 = None
                for bc2_len in range(max(self._bc_lengths) + 1, min(self._bc_lengths) - 2, -1):  # padded range from max length+1 to min length-1
                    if bc1_end + bc2_len > len(read):
                        continue
                    possible_bc2 = read[bc1_end:bc1_end+bc2_len]
                    bc2, bc2_corrections = self._cached_bc2_search(possible_bc2, max_mismatches)
                    if bc2 is None:
                        continue
                    if bc2_corrections < best_corrections or (bc2_corrections == best_corrections and bc2_len > best_len):
                        best_corrections = bc2_corrections
                        best_len = bc2_len
                        best_bc2 = bc2
                        if best_corrections == 0:  # Can't get better than this
                            break
                if best_bc2 is not None:
                    return bc1 + best_bc2, best_corrections
                return None, -1  # No match found
            case (False, True):  # Only bc2 valid, search for bc1 with mismatches
                bc2_start = bc2_range[0]
                bc2 = read[bc2_range[0]:bc2_range[1]]
                # Search for bc1 with mismatches
                best_len = -1
                best_corrections = 1e6
                best_bc1 = None
                for bc1_len in range(max(self._bc_lengths) + 1, min(self._bc_lengths) - 2, -1):  # padded range from max length+1 to min length-1
                    if bc2_start - bc1_len < 0:
                        continue
                    possible_bc1 = read[bc2_start-bc1_len:bc2_start]
                    bc1, bc1_corrections = self._cached_bc1_search(possible_bc1, max_mismatches)
                    if bc1 is None:
                        continue
                    if bc1_corrections < best_corrections or (bc1_corrections == best_corrections and bc1_len > best_len):
                        best_corrections = bc1_corrections
                        best_len = bc1_len
                        best_bc1 = bc1
                        if best_corrections == 0:  # Can't get better than this
                            break
                if best_bc1 is not None:
                    return best_bc1 + bc2, best_corrections
                return None, -1  # No match found
            case (False, False):
                # Full fuzzy search
                # Search with padding
                best_bc = None
                best_bc_corrections = None
                best_length = None
                search_ranges = self.possible_pairing_ranges(len(read), padding=1)
                for (possible_bc1_start, possible_bc1_end), (possible_bc2_start, possible_bc2_end) in search_ranges:
                    if possible_bc2_end > len(read) or possible_bc1_start < 0:
                        continue
                    possible_bc1 = read[possible_bc1_start:possible_bc1_end]
                    possible_bc2 = read[possible_bc2_start:possible_bc2_end]
                    # Search for a valid sequence
                    bc1, bc1_corrections = self._cached_bc1_search(possible_bc1, max_mismatches)
                    if bc1 is None:
                        continue
                    remaining_mismatches = max_mismatches - bc1_corrections
                    if remaining_mismatches < 0:
                        continue
                    bc2, bc2_corrections = self._cached_bc2_search(possible_bc2, remaining_mismatches)
                    if bc2 is None:
                        continue
                    if bc1_corrections + bc2_corrections > max_mismatches:
                        continue
                    combined_corrections = (bc1_corrections + bc2_corrections)
                    combined_length = possible_bc1_end - possible_bc1_start + 1
                    if best_bc is None or best_bc_corrections < combined_corrections or (combined_length > best_length and best_bc_corrections == combined_corrections):
                        best_bc = bc1 + bc2
                        best_length = combined_length
                        best_bc_corrections = combined_corrections
                        if best_bc_corrections == 0:  # Can't get better than this
                            break
                if best_bc is not None and best_bc in self._barcode_coordinates:
                    return best_bc, best_bc_corrections
        # No matches found, return None
        return None, -1


class ProbeParser:

    def __init__(self,
                 lhs_seqs: list[str],
                 rhs_seqs: list[str],
                 names: list[str],
                 tech: TechnologyFormatInfo,
                 probe_bcs: Optional[list[int]] = None,  # Indices of the probe barcodes corresponding to the sequences (1-indexed)
                 allow_indels: bool = False):
        self.lhs_seqs = lhs_seqs
        self.rhs_seqs = rhs_seqs
        self.names = names

        # Deduplicate sequences and map back to pairings
        self.deduped_lhs = defaultdict(set)
        self.deduped_rhs = defaultdict(set)
        self.lhs_lens = list(sorted(set(len(seq) for seq in lhs_seqs), reverse=True))
        self.rhs_lens = list(sorted(set(len(seq) for seq in rhs_seqs), reverse=True))
        self.lhs_lens_differ = len(self.lhs_lens) > 1
        self.rhs_lens_differ = len(self.rhs_lens) > 1
        for i, (lhs, rhs) in enumerate(zip(lhs_seqs, rhs_seqs)):
            self.deduped_lhs[lhs].add(i)
            self.deduped_rhs[rhs].add(i)

        # Create prefix tries for fast searching
        # self.lhs_trie = PrefixTrie(list(self.deduped_lhs.keys()), allow_indels=allow_indels)
        # self.rhs_trie = PrefixTrie(list(self.deduped_rhs.keys()), allow_indels=allow_indels)
        self.lhs_trie, self._lhs_trie_name = create_shared_trie(list(self.deduped_lhs.keys()), allow_indels=allow_indels)
        self.rhs_trie, self._rhs_trie_name = create_shared_trie(list(self.deduped_rhs.keys()), allow_indels=allow_indels)
        self.constant_seq_trie = None
        self._constant_seq_trie_name = None
        self.probe_bc_trie = None
        self._probe_bc_trie_name = None

        self.has_constant_seq = tech.has_constant_sequence
        if self.has_constant_seq:
            self.constant_seq = tech.constant_sequence
            self.constant_seq_start = tech.constant_sequence_start
            # Create a trie for the constant sequence
            # self.constant_seq_trie = PrefixTrie([self.constant_seq], allow_indels=allow_indels)
            self.constant_seq_trie, self._constant_seq_trie_name = create_shared_trie([self.constant_seq], allow_indels=allow_indels)
        else:
            self.constant_seq = None
            self.constant_seq_start = None

        self.probe_bcs = probe_bcs
        self.has_probe_bc = tech.has_probe_barcode and probe_bcs is not None and len(probe_bcs) > 0
        if self.has_probe_bc:
            self.probe_bc_start = tech.probe_barcode_start
            self.probe_bc_length = tech.probe_barcode_length
            # self.probe_bc_trie = PrefixTrie([tech.probe_barcodes[i-1] for i in probe_bcs], allow_indels=allow_indels)
            self.probe_bc_trie, self._probe_bc_trie_name = create_shared_trie([tech.probe_barcodes[i-1] for i in probe_bcs], allow_indels=allow_indels)
        else:
            self.probe_bc_start = None
            self.probe_bc_length = None

    def __getstate__(self):
        state = dict()
        state['_lhs_trie_name'] = self._lhs_trie_name
        state['_rhs_trie_name'] = self._rhs_trie_name
        state['_constant_seq_trie_name'] = self._constant_seq_trie_name
        state['_probe_bc_trie_name'] = self._probe_bc_trie_name
        state['lhs_seqs'] = self.lhs_seqs
        state['rhs_seqs'] = self.rhs_seqs
        state['names'] = self.names
        state['deduped_lhs'] = self.deduped_lhs
        state['deduped_rhs'] = self.deduped_rhs
        state['lhs_lens'] = self.lhs_lens
        state['rhs_lens'] = self.rhs_lens
        state['lhs_lens_differ'] = self.lhs_lens_differ
        state['rhs_lens_differ'] = self.rhs_lens_differ
        state['has_constant_seq'] = self.has_constant_seq
        state['constant_seq'] = self.constant_seq
        state['constant_seq_start'] = self.constant_seq_start
        state['probe_bcs'] = self.probe_bcs
        state['probe_bc_start'] = self.probe_bc_start
        state['probe_bc_length'] = self.probe_bc_length
        state['has_probe_bc'] = self.has_probe_bc
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.lhs_trie = load_shared_trie(self._lhs_trie_name)
        self.rhs_trie = load_shared_trie(self._rhs_trie_name)
        if self.has_constant_seq:
            self.constant_seq_trie = load_shared_trie(self._constant_seq_trie_name)
        if self.has_probe_bc:
            self.probe_bc_trie = load_shared_trie(self._probe_bc_trie_name)

    def _compute_max_distance(self, string_length: int, max_mismatches: int) -> int:
        # Compute the maximum distance allowed based on the length of the string
        return max(1, (string_length * max_mismatches) // 10)

    @functools.lru_cache(maxsize=250_000)
    def parse_probe(self, read2: str,
                    max_mismatches: int,
                    skip_constant_seq: bool = False,
                    flexible_start: bool = False) -> tuple[Optional[int], Optional[int], Optional[int], Optional[str], Optional[str], list[ReadProcessState]]:
        """
        From the given read2 sequence, parse the probe, gapfill, probe bc, and process states.
        :param read2: The R2 sequence to parse.
        :param max_mismatches: The maximum number of mismatches to allow per 10bp.
        :param skip_constant_seq: If True, do not filter out reads that do not have the constant sequence.
        :param flexible_start: If True, allow for flexible start positions for the LHS sequence (i.e. not anchored to the start of the read).
        :return: The probe index, gapfill sequence, gapfill_start, gapfill_end, probe barcode, and process states.
        """
        state = [ReadProcessState.TOTAL_READS]
        if flexible_start:
            lhs, lhs_corrections, lhs_start, lhs_end = self.lhs_trie.search_substring(read2, self._compute_max_distance(max(self.lhs_lens), max_mismatches))
            if lhs is None:
                state.append(ReadProcessState.FILTERED_NO_LHS)
                return None, None, None, None, None, state
            if lhs_corrections > 0:
                state.append(ReadProcessState.CORRECTED_LHS)
            # Remove the lhs from the read
            read2 = read2[lhs_end:]
        else:
            # First, identify if the LHS is present
            if self.lhs_lens_differ:
                # Try matching from longest to shortest
                lhs = None
                for lhs_len in self.lhs_lens:
                    if lhs_len > len(read2):
                        continue
                    possible_lhs = read2[0:lhs_len]
                    lhs, lhs_corrections = self.lhs_trie.search(possible_lhs, self._compute_max_distance(len(possible_lhs), max_mismatches))
                    if lhs is not None:
                        if lhs_corrections > 0:
                            state.append(ReadProcessState.CORRECTED_LHS)
                        break
                if lhs is None:
                    state.append(ReadProcessState.FILTERED_NO_LHS)
                    return None, None, None, None, None, state
            else:
                # Single search
                possible_lhs = read2[0:self.lhs_lens[0]]
                lhs, lhs_corrections = self.lhs_trie.search(possible_lhs, self._compute_max_distance(len(possible_lhs), max_mismatches))
                if lhs is None:
                    state.append(ReadProcessState.FILTERED_NO_LHS)
                    return None, None, None, None, None, state
                if lhs_corrections > 0:
                    state.append(ReadProcessState.CORRECTED_LHS)

            # Remove the lhs from the read
            read2 = read2[len(lhs):]

        # Next, check for constant sequence
        probe_bc = ""
        rhs_end = None
        if self.has_constant_seq:
            search_space = read2[max(self.rhs_lens) + self.constant_seq_start:]
            # Attempt to find the constant sequence
            found_string, corrections, constant_seq_start_pos, constant_seq_end_pos = self.constant_seq_trie.search_substring(search_space, self._compute_max_distance(len(self.constant_seq), max_mismatches))

            if found_string is None:  # Could not find constant seq. Possible that it was cutoff, so lets find the best prefix match
                # We need at least 1/2 of the constant sequence to consider it a match
                min_constant_len = len(self.constant_seq) // 2
                found_string, constant_seq_start_pos, match_len = self.constant_seq_trie.longest_prefix_match(search_space, min_constant_len, self._compute_max_distance(len(self.constant_seq), max_mismatches))

                if found_string is None and not skip_constant_seq:
                    state.append(ReadProcessState.FILTERED_NO_CONSTANT)
                    return None, None, None, None, None, state
                else:
                    constant_seq_end_pos = constant_seq_start_pos + match_len
            if found_string is not None:
                # Adjust the positions to be relative to the original read2
                constant_seq_start_pos += len(read2) - len(search_space)
                constant_seq_end_pos += len(read2) - len(search_space)
                rhs_end = constant_seq_start_pos - self.constant_seq_start

                if self.has_probe_bc:
                    possible_probe_bc = read2[constant_seq_end_pos + self.probe_bc_start:constant_seq_end_pos + self.probe_bc_start + self.probe_bc_length]
                    if len(possible_probe_bc) == 0:
                        state.append(ReadProcessState.FILTERED_NO_PROBE_BARCODE)
                        return None, None, None, None, None, state
                    probe_bc, probe_bc_corrections = self.probe_bc_trie.search(possible_probe_bc, self._compute_max_distance(len(possible_probe_bc), max_mismatches))
                    if probe_bc is None:
                        if len(possible_probe_bc) < self.probe_bc_length:  # Try prefix search
                            probe_bc, probe_bc_start, match_len = self.probe_bc_trie.longest_prefix_match(possible_probe_bc, min_match_length=self.probe_bc_length//2, correction_budget=self._compute_max_distance(len(possible_probe_bc), max_mismatches))
                        if probe_bc is None:
                            state.append(ReadProcessState.FILTERED_NO_PROBE_BARCODE)
                            return None, None, None, None, None, state
        else:  # No need to search for constant sequence and probe bc
            rhs_end = len(read2)

        # Now we try to match the RHS
        read2 = read2[:rhs_end]

        # Short circuit cases:
        if len(read2) < min(self.rhs_lens):  # Impossible to have the full RHS
            # Search for the RHS with a prefix match
            rhs, rhs_start, match_len = self.rhs_trie.longest_prefix_match(read2, min_match_length=min(self.rhs_lens)//2, correction_budget=self._compute_max_distance(max(self.rhs_lens), max_mismatches))
            if rhs is None:
                state.append(ReadProcessState.FILTERED_NO_RHS)
                return None, None, None, None, None, state
            if rhs != read2[rhs_start:rhs_start+match_len]:
                state.append(ReadProcessState.CORRECTED_RHS)
        else:
            rhs, rhs_corrections, rhs_start, rhs_end = self.rhs_trie.search_substring(read2, self._compute_max_distance(max(self.rhs_lens), max_mismatches))
            if rhs is None:
                state.append(ReadProcessState.FILTERED_NO_RHS)
                return None, None, None, None, None, state
            if rhs_corrections > 0:
                state.append(ReadProcessState.CORRECTED_RHS)

        # extract the gapfill
        gapfill = read2[:rhs_start]

        possible_lhs_indices = self.deduped_lhs[lhs]
        possible_rhs_indices = self.deduped_rhs[rhs]

        possible_indices = possible_lhs_indices.intersection(possible_rhs_indices)
        if len(possible_indices) == 0:
            state.append(ReadProcessState.FILTERED_NO_RHS)
            return None, None, None, None, None, state

        probe_idx = possible_indices.pop()

        return probe_idx, gapfill, len(lhs), len(lhs) + rhs_start, probe_bc, state


def read_barcodes(output_dir: Path) -> pd.DataFrame:
    """
    Read the list of cell barcodes.
    :param output_dir: The output directory.
    :return: The list of cell barcodes.
    """
    # FIXME: TURN INTO DATAFRAME WITH SPATIAL COORDS
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
    if (output_dir / "barcodes.tsv").exists():
        return pd.read_table(output_dir / "barcodes.tsv")
    elif (output_dir / "barcodes.tsv.gz").exists():
        return pd.read_table(output_dir / "barcodes.tsv.gz")
    else:
        raise FileNotFoundError("Barcodes file not found.")


# Create a file writer handler that wraps gzip and NamedTemporaryFile
class GzipNamedTemporaryFile:

    def __init__(self):
        self.temp_file = NamedTemporaryFile(mode="w+b", delete=False)
        self.gzip_file = gzip.GzipFile(fileobj=self.temp_file, mode="w")
        # Note that GzipFile only supports binary mode:
        self.gzip_file = io.TextIOWrapper(self.gzip_file)

    def __enter__(self):
        self.temp_file.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.gzip_file.close()
        return self.temp_file.__exit__(exc_type, exc_val, exc_tb)

    @property
    def name(self):
        return self.temp_file.name

    def write(self, data: str):
        self.gzip_file.write(data)


def maybe_gzip(file: Path | None, mode: Literal["r"] | Literal["w"] = "r"):
    """
    Return a file handle. If the file is gzipped, then we will use gzip.open, otherwise we will use open.
    :param file: The file path. If None, this will return a temporary file.
    :param mode: The mode.
    :return: The file handle.
    """
    if file is None:
        if mode == 'w':
            return GzipNamedTemporaryFile()  # context manager
        raise ValueError("read mode requires a real path")

    file = Path(file)
    if mode == 'r':
        if "gz" in file.suffix:
            return gzip.open(file, 'rt')
        else:
            return open(file, mode)
    else:
        if "gz" in file.suffix:
            return gzip.open(file, 'wt')
        else:
            return open(file, mode)


def compile_flatfile(manifest_df: pd.DataFrame, probe_reads_file: str, barcode_list: list[str], plex: int, output: str):
    """
    Flatten giftwrap data to a human readable tsv-based format.
    :param manifest_df: The manifest dataframe.
    :param probe_reads_file: The probe reads file.
    :param barcode_list: The index to barcode mapping. If a barcode is not present here, it will be dropped.
    :param plex: The plex number.
    :param output: The output with the following columns: cell barcode, LHS, RHS, probe_call, gapfill, pcr duplicates.
        Where each row represents an individual umi.
    """
    with maybe_gzip(probe_reads_file, 'r') as input_file, maybe_gzip(output, 'w') as output_file:
        # Skip the header
        next(input_file)
        output_file.write(f"cell_barcode\tlhs_probe\trhs_probe\tcalled_probe\tgapfill\tpcr_duplicates\tpercent_supporting\tumi\n")
        for line in input_file:
            split = line.strip().split("\t")
            if len(split) != 6:
                continue
            cell_idx, probe_idx, probe_bc_idx, umi, gapfill, umi_count, percent_supporting = line.strip().split("\t")
            if int(cell_idx) > len(barcode_list) or int(probe_bc_idx) != plex:
                continue

            cell_barcode = barcode_list[int(cell_idx)]
            probe = manifest_df[manifest_df["index"] == int(probe_idx)].iloc[0]
            if 'was_defined' in probe:
                if probe['was_defined']:
                    lhs_probe = probe['name']
                    rhs_probe = probe['name']
                else:
                    lhs_probe = probe['name'].split("/")[0]
                    rhs_probe = probe['name'].split("/")[1]
            else:
                lhs_probe = probe['name']
                rhs_probe = probe['name']

            output_file.write(f"{cell_barcode}\t{lhs_probe}\t{rhs_probe}\t{probe_bc_idx}\t{gapfill}\t{umi_count}\t{percent_supporting}\t{umi}\n")


def _parse_barcodes_tsv(filepath: Path) -> np.ndarray[str]:
    """
    Parse a barcodes.tsv file.
    :param filepath: The path to the barcodes.tsv(.gz) file.
    :return: A list of barcodes.
    """
    return pd.read_csv(filepath, sep="\t", header=None, compression='gzip' if filepath.suffix == '.gz' else None).iloc[:, 0].str.split("-").str[0].to_numpy(dtype=str)


def _parse_molecule_info_h5(filepath: Path) -> np.ndarray[str]:
    """
    Parse a VisiumHD molecule_info.h5 file.
    :param filepath: The path to the molecule_info.h5 file.
    :return: A list of barcodes.
    """
    with h5py.File(filepath, "r") as f:
        barcodes = pd.Series(f['barcodes'].asstr()[()], dtype=str)

    # Remove -1 from the barcodes
    return barcodes.str.split("-").str[0].to_numpy(dtype=str)


def _parse_filtered_feature_bc_matrix_h5(filepath: Path) -> np.ndarray[str]:
    """
    Parse a filtered_feature_bc_matrix.h5 file.
    :param filepath: The path to the filtered_feature_bc_matrix.h5 file.
    :return: A list of barcodes.
    """
    with h5py.File(filepath, "r") as f:
        barcodes = pd.Series(f['matrix']['barcodes'].asstr()[()], dtype=str)

    # Remove -1 from the barcodes
    return barcodes.str.split("-").str[0].to_numpy(dtype=str)


def read_wta(
        input_path: Path,
        barcodes_only: bool = False,
        fallback_to_barcodes: bool = False,
) -> Union[ad.AnnData, np.ndarray[str]]:
    """
    Read a WTA file and return the cells processed by cellranger or spaceranger.
    :param input_path: The path to the WTA file.
    :param barcodes_only: If true, return only the barcodes.
    :param fallback_to_barcodes: If true, fallback to barcodes if scanpy is not available regardless.
    :return: The cells processed by cellranger. An AnnData object if barcodes_only is False, otherwise a DataFrame.

    Note that this prefers to parse outputs using scanpy as it will be the most robust, but if scanpy is not available,
        we will try to parse outputs to extract just cell barcodes.
    """
    # Check if this is cellranger of spaceranger according to the file structure
    if barcodes_only:
        if input_path.is_dir():
            # Check if square_002um appears in the directory structure
            if "square_002um" in str(input_path):  # Pointing to the binned output
                if "filtered_feature_bc_matrix" in str(input_path):
                    return _parse_barcodes_tsv(input_path / "barcodes.tsv.gz")
                else:
                    return _parse_barcodes_tsv(input_path / "filtered_feature_bc_matrix" / "barcodes.tsv.gz")
            elif (input_path / "spatial").exists():  # Pointing to the spatial output base directory
                return _parse_barcodes_tsv(input_path / "binned_outputs" / "square_002um" / "filtered_feature_bc_matrix" / "barcodes.tsv.gz")
            elif (input_path / "outs" / "binned_outputs" / "square_002um").exists():
                return _parse_barcodes_tsv(input_path / "outs" / "binned_outputs" / "square_002um" / "filtered_feature_bc_matrix" / "barcodes.tsv.gz")
            elif (input_path / "binned_outputs" / "square_002um").exists():
                return _parse_barcodes_tsv(input_path / "binned_outputs" / "square_002um" / "filtered_feature_bc_matrix" / "barcodes.tsv.gz")
            elif (input_path / "square_002um").exists():
                return _parse_barcodes_tsv(input_path / "square_002um" / "filtered_feature_bc_matrix" / "barcodes.tsv.gz")
            else: # Assume cell ranger output
                return _parse_barcodes_tsv(input_path / "barcodes.tsv.gz")
        else:
            base_filename = input_path.name
            if (base_filename == 'molecule_info.h5' and (input_path.parent / 'spatial').exists()):  # Given a molecule_info.h5 file in a spatial directory
                return _parse_molecule_info_h5(input_path)
            elif (base_filename == "filtered_feature_bc_matrix.h5" and (input_path.parent / 'spatial').exists()):
                return _parse_filtered_feature_bc_matrix_h5(input_path)
            elif base_filename == "sample_filtered_feature_bc_matrix.h5":
                return _parse_filtered_feature_bc_matrix_h5(input_path)
            elif base_filename == "sample_molecule_info.h5":
                return _parse_molecule_info_h5(input_path)
            else:  # Try parsing successively
                try:
                    return _parse_barcodes_tsv(input_path)
                except:
                    try:
                        return _parse_filtered_feature_bc_matrix_h5(input_path)
                    except:
                        return _parse_molecule_info_h5(input_path)
        raise FileNotFoundError("Barcodes file not found.")

    try:
        import scanpy as sc
    except:
        if not barcodes_only and not fallback_to_barcodes:
            print("Scanpy not found. Please install it to use the cellranger output.")
            return
        elif fallback_to_barcodes:
            return read_wta(input_path, barcodes_only=True)

    if input_path.is_dir():
        adata = sc.read_10x_mtx(input_path)
    else:
        adata = sc.read_10x_h5(input_path)

    if barcodes_only:
        return adata.obs_names.values
    else:
        return adata


def sort_tsv_file(file: Path, columns: list[int], cores: int):
    """
    Sort a written tsv file in-place. Will either use a single core or multiple cores depending on the cores argument.
        Note, this will attempt to defer to the unix sort command if cores is > 1.
    :param file: The file. May be gzipped.
    :param columns: The columns indices to sort by.
    :param cores: The number of cores to use.
    """
    if cores > 1:
        # Check for the sort command
        sort_avail = shutil.which("sort")
        if sort_avail:
            # Use the unix sort command
            # First move to a temporary file
            os.rename(file, file.with_suffix(".tmp"))
            # Then sort (Ignore locale for all commands for speed)
            sort_command = "export LC_ALL=C; "
            # First open the file
            if 'gz' in file.suffix:
                sort_command += f"zcat {file.with_suffix('.tmp')} | "
            else:
                sort_command += f"cat {file.with_suffix('.tmp')} | "
            # Note that we need to skip the first line: https://unix.stackexchange.com/a/11857
            sort_command += '(IFS= read -r REPLY; printf "%s\\n" "$REPLY"; '
            # Then sort
            sort_command += f"sort -t \"$(printf '\\t')\" --parallel={cores} --numeric-sort"
            # Note that sort doesn't parallelize piped input since it assumes its a small file so we will give it a
            # large buffer size (1 GB per core)
            sort_command += f" --buffer-size={cores}G"
            # Note that we need to add 1 to the column index since sort is 1-indexed
            for col in columns:
                sort_command += f" -k{col + 1},{col + 1}"
            # Close the parenthesis
            sort_command += ")"

            # if the file is gzipped, then we need to gzip the output
            if ".gz" in file.suffix:
                sort_command += f" | gzip > {file}"
            else:
                sort_command += f" > {file}"

            result = subprocess.run(sort_command, shell=True)
            if result.returncode != 0:
                # Move the file back
                os.rename(file.with_suffix(".tmp"), file)
                raise RuntimeError("Failed to sort the file.")
            # Delete the backup file
            os.remove(file.with_suffix(".tmp"))
            return

    # Not able to use the sort command so fallback to python
    df = pd.read_table(file, sep="\t", compression="gzip" if "gz" in file.suffix else None)
    df = df.sort_values(df.columns[columns].tolist())
    df.to_csv(file, sep="\t", index=False, compression="gzip" if "gz" in file.suffix else None)


def filter_h5_file_by_barcodes(input_file: Path, output_file: Path, barcodes_list: np.ndarray[str], pad_matrix: bool = True):
    """
    Given a counts h5 file and a list of barcodes, filter the barcodes to only include the ones in the list.
    :param input_file: The input h5 file.
    :param output_file: The output h5 file.
    :param barcodes_list: The barcodes list.
    :param pad_matrix: Whether to pad the matrix with zeros if there are barcodes provided that don't exist in the file.
    """
    # First, copy the file
    shutil.copy(input_file, output_file)

    # Convert barcodes_list to a set for O(1) lookups
    barcodes_set = set(barcodes_list)

    # Then open the file
    with h5py.File(output_file, 'r+') as f:
        # Read barcodes once
        barcodes_array = f['matrix']['barcode'][:]
        barcodes = np.char.decode(barcodes_array, 'utf-8') if barcodes_array.dtype.kind == 'S' else barcodes_array.astype(str)

        # Use vectorized numpy operations for finding indices
        # Create a boolean mask for matching barcodes
        mask = np.isin(barcodes, list(barcodes_set))
        barcode_indices = np.where(mask)[0]

        # Check if we need to filter the data
        if len(barcode_indices) == len(barcodes):
            return  # Equal size, no point in filtering

        if len(barcode_indices) == 0:
            raise ValueError("No barcodes found in the file.")

        # Calculate padded barcodes only if needed
        padded_barcodes = []
        if pad_matrix and len(barcodes_set) > len(barcode_indices):
            existing_barcodes = set(barcodes)
            padded_barcodes = np.array([bc for bc in barcodes_list if bc not in existing_barcodes], dtype='S')
            print(f"Padding {len(padded_barcodes)} unseen cells with zeroes.")

        # Filter the data
        filtered_barcodes = barcodes[barcode_indices]
        del f['matrix']['barcode']

        # Convert to bytes only once if needed
        if filtered_barcodes.dtype.kind != 'S':
            filtered_barcodes = np.array(filtered_barcodes, dtype='S')

        # Create the new dataset
        if len(padded_barcodes) > 0:
            if padded_barcodes.dtype.kind != 'S':
                padded_barcodes = np.array(padded_barcodes, dtype='S')
            combined_barcodes = np.concatenate([filtered_barcodes, padded_barcodes])
        else:
            combined_barcodes = filtered_barcodes

        f['matrix'].create_dataset("barcode", data=combined_barcodes, compression='gzip')

        # Filter cell_index if it exists
        if 'cell_index' in f['matrix']:
            cell_indices = f['matrix']['cell_index'][:]
            del f['matrix']['cell_index']
            # For padded barcodes, we'll use -1 as a placeholder for missing original indices
            if len(padded_barcodes) > 0:
                padded_cell_indices = np.full(len(padded_barcodes), -1, dtype=np.int32)
                combined_indices = np.concatenate([cell_indices[barcode_indices], padded_cell_indices])
            else:
                combined_indices = cell_indices[barcode_indices]
            f['matrix'].create_dataset("cell_index", data=combined_indices, compression='gzip')

        # Process each sparse matrix layer
        for layer_name in ['data', 'total_reads', 'percent_supporting']:
            data = read_sparse_matrix(f['matrix'], layer_name)
            # Efficiently extract rows
            data = data[barcode_indices, :]

            # Add padding only if needed
            if len(padded_barcodes) > 0:
                # Create empty sparse matrix only once with correct dimensions
                padding = scipy.sparse.csr_matrix((len(padded_barcodes), data.shape[1]), dtype=data.dtype)
                data = scipy.sparse.vstack([data, padding])

            del f['matrix'][layer_name]
            write_sparse_matrix(f['matrix'], layer_name, data)

        # If all_pcr_thresholds is present, filter it as well
        if 'max_pcr_duplicates' in f.attrs:
            max_pcr_dups = int(f.attrs['max_pcr_duplicates'])
            if max_pcr_dups > 1:
                all_pcr_grp = f["pcr_thresholded_counts"]
                for pcr_dup in range(1, max_pcr_dups):
                    layer_name = f"pcr{pcr_dup}"
                    if layer_name in all_pcr_grp:
                        data = read_sparse_matrix(all_pcr_grp, layer_name)
                        data = data[barcode_indices, :]

                        # Add padding only if needed
                        if len(padded_barcodes) > 0:
                            # Create empty sparse matrix only once with correct dimensions
                            padding = scipy.sparse.csr_matrix((len(padded_barcodes), data.shape[1]), dtype=data.dtype)
                            data = scipy.sparse.vstack([data, padding])

                        del all_pcr_grp[layer_name]
                        write_sparse_matrix(all_pcr_grp, layer_name, data)

        # Process cell metadata more efficiently
        if 'cell_metadata' in f:
            obs_meta_columns = f['cell_metadata']['columns'][:].astype(str)

            # Create a dictionary to hold column data
            meta_dict = {}
            for col in obs_meta_columns:
                values = f['cell_metadata'][col][:]
                if col == 'barcode':
                    if values.dtype.kind == 'S':
                        values = np.char.decode(values, 'utf-8')
                    meta_dict[col] = values
                else:
                    try:
                        meta_dict[col] = values
                    except:
                        meta_dict[col] = np.zeros_like(values, dtype=int)

            # Create filtered DataFrame
            obs_meta_df = pd.DataFrame(meta_dict)
            if 'barcode' in obs_meta_df.columns:
                obs_meta_df.set_index('barcode', inplace=True)
                filtered_meta = obs_meta_df.loc[barcodes[barcode_indices]].reset_index()
            else:
                filtered_meta = obs_meta_df.iloc[barcode_indices].reset_index(drop=True)

            # Add padding if needed
            if len(padded_barcodes) > 0:
                pad_dict = {col: ([pd.NA] * len(padded_barcodes)) for col in filtered_meta.columns if col != 'barcode'}
                if 'barcode' in filtered_meta.columns:
                    pad_dict['barcode'] = padded_barcodes
                pad_df = pd.DataFrame(pad_dict)
                filtered_meta = pd.concat([filtered_meta, pad_df], ignore_index=True)

            # Write back to file
            del f['cell_metadata']
            cell_metadata_grp = f.create_group('cell_metadata')
            cell_metadata_grp.create_dataset('columns', data=np.array(filtered_meta.columns, dtype='S'), compression='gzip')

            for col in filtered_meta.columns:
                values = filtered_meta[col].values
                # Convert to appropriate type
                if not np.issubdtype(values.dtype, np.number):
                    values = np.array(values, dtype='S')
                cell_metadata_grp.create_dataset(col, data=values, compression='gzip')

        # Update cell count
        if 'n_cells' in f.attrs:
            del f.attrs['n_cells']
            f.attrs['n_cells'] = len(combined_barcodes)

        # Done


def filter_h5_file_by_pcr_dups(probe_reads_file: Path, counts_input: Path,
                               counts_output: Path, reads_per_gapfill: int,
                               probe_bc: int):
    """
    Filter an h5 file by removing UMIs with low PCR duplicate counts. This function rebuilds the count matrices
    using only the UMIs that meet the specified threshold.

    :param probe_reads_file: Path to the probe reads file containing individual UMI records.
    :param counts_input: Input h5 counts file to be filtered.
    :param counts_output: Path to write the filtered h5 counts file.
    :param reads_per_gapfill: Minimum number of PCR duplicates (reads) required for a UMI to be retained.
    :param probe_bc: The probe barcode index to filter on.
    """
    if not counts_output.exists():
        shutil.copy(counts_input, counts_output)

    # Open the new file in read/write mode to modify it in-place.
    with h5py.File(counts_output, 'r+') as f:
        original_cell_indices = f['matrix']['cell_index'][:]
        original_probe_indices = f['matrix']['probe_index'][:]

        cell_idx_to_row = {cell_id: i for i, cell_id in enumerate(original_cell_indices)}
        probe_idx_to_col = {probe_id: j for j, probe_id in enumerate(original_probe_indices)}
        subtracted_counts = defaultdict(lambda: {'data': 0, 'total_reads': 0})

        with maybe_gzip(probe_reads_file, 'r') as pf:
            next(pf)  # Skip header
            for line in pf:
                parts = line.rstrip().split('\t')
                # Ensure the line has enough columns to parse
                if len(parts) < 6:
                    continue

                cell_idx, probe_idx, probe_barcode, _, _, umi_count = parts[:6]
                if int(probe_barcode) != probe_bc:
                    continue
                cell_idx = int(cell_idx)
                probe_idx = int(probe_idx)
                umi_count = int(umi_count)

                # Apply the filter based on PCR duplicates
                if umi_count < reads_per_gapfill:
                    # Find the corresponding matrix row and column for this UMI
                    row = cell_idx_to_row.get(cell_idx)
                    col = probe_idx_to_col.get(probe_idx)

                    # Only proceed if the cell and probe exist in the target H5 file
                    if row is not None and col is not None:
                        # Increment the unique UMI count for this cell/probe pair
                        subtracted_counts[(row, col)]['data'] += 1
                        # Add the PCR duplicates to the total reads count
                        subtracted_counts[(row, col)]['total_reads'] += umi_count

        if not subtracted_counts:  # No changes needed
            print("All UMIs passed the filter; no changes made to the counts file.")
            return

        # Read matrices and convert to more efficient format for modifications
        original_data = read_sparse_matrix(f['matrix'], 'data').tolil()
        original_total_reads = read_sparse_matrix(f['matrix'], 'total_reads').tolil()
        original_percent_supporting = read_sparse_matrix(f['matrix'], 'percent_supporting').tolil()

        for (r, c), counts in subtracted_counts.items():
            # Convert to int to avoid unsigned integer overflow, then apply max(0, ...)
            original_data[r, c] = max(0, int(original_data[r, c]) - counts['data'])
            original_total_reads[r, c] = max(0, int(original_total_reads[r, c]) - counts['total_reads'])
            if original_total_reads[r, c] > 0:
                original_percent_supporting[r, c] = (original_data[r, c] / original_total_reads[r, c]) * 100
            else:
                original_percent_supporting[r, c] = 0.0

        # Convert back to CSR format for storage
        original_data = original_data.tocsr()
        original_total_reads = original_total_reads.tocsr()
        original_percent_supporting = original_percent_supporting.tocsr()
        # Save the modified matrices back to the file
        print("Writing filtered matrices back to HDF5 file...")
        for layer_name, matrix in zip(
            ['data', 'total_reads', 'percent_supporting'],
            [original_data, original_total_reads, original_percent_supporting]
        ):
            if layer_name in f['matrix']:
                del f['matrix'][layer_name]
            write_sparse_matrix(f['matrix'], layer_name, matrix)


def write_sparse_matrix(grp: h5py.Group, name: str, sp_matrix):
    """
    Write a sparse matrix to a group.
    :param grp: The group.
    :param name: The name of the dataset.
    :param sp_matrix: The sparse matrix.
    """
    if not scipy.sparse.isspmatrix_csr(sp_matrix):
        sp_matrix = sp_matrix.tocsr()

    matrix_grp = grp.create_group(name)
    matrix_grp.create_dataset("data", data=sp_matrix.data, compression='gzip', shuffle=True)
    matrix_grp.create_dataset("indices", data=sp_matrix.indices, compression='gzip', shuffle=True)
    matrix_grp.create_dataset("indptr", data=sp_matrix.indptr, compression='gzip', shuffle=True)
    # Shape
    matrix_grp.attrs['shape'] = sp_matrix.shape


def read_sparse_matrix(grp: h5py.Group, name: str) -> scipy.sparse.csr_matrix:
    """
    Read a sparse matrix from a group.
    :param grp: The group.
    :param name: The name of the dataset.
    :return: The sparse matrix.
    """
    matrix_grp = grp[name]
    shape = matrix_grp.attrs['shape']
    return scipy.sparse.csr_matrix((matrix_grp['data'], matrix_grp['indices'], matrix_grp['indptr']), shape=shape)


def read_h5_file(filename: str) -> ad.AnnData:
    """
    Read a generated h5 file and return an AnnData object.
    :param filename: The filename.
    :return: The AnnData object.
    """
    with h5py.File(filename, 'r') as f:
        X = read_sparse_matrix(f['matrix'], 'data')
        layers = {
            'total_reads': read_sparse_matrix(f['matrix'], 'total_reads'),  # Total umis encountered
            'percent_supporting': read_sparse_matrix(f['matrix'], 'percent_supporting'),  # Avg percent of umis supporting the gapfill call
        }
        var_df = pd.DataFrame({
            'probe': f['matrix']['probe'][:, 0].astype(str),
            'gapfill': f['matrix']['probe'][:, 1].astype(str),
        })

        # Add original probe indices if available
        if 'probe_index' in f['matrix']:
            var_df['probe_index'] = f['matrix']['probe_index'][:].astype(int)

        obs_df = pd.DataFrame({
            'barcode': f['matrix']['barcode'][:].astype(str),
        }).set_index('barcode')

        # Add original cell indices if available
        if 'cell_index' in f['matrix']:
            obs_df['cell_index'] = f['matrix']['cell_index'][:].astype(int)

        # Read the obs metadata
        obs_meta_columns = f['cell_metadata']['columns'][:].astype(str)
        obs_meta_df = dict()
        for column in obs_meta_columns:
            values = f['cell_metadata'][column][:]
            if column == 'barcode':
                values = values.astype(str)
            else:
                try:
                    values = values.astype(int)  # Most metadata are ints
                except:  # If that doesn't work, try string
                    try:
                        values = values.astype(str)
                    except:
                        values = np.zeros_like(values, dtype=int)  # Give up
            obs_meta_df[column] = values
        obs_meta_df = pd.DataFrame(obs_meta_df).set_index("barcode")

        obs_df = obs_df.merge(obs_meta_df, on='barcode', how='left')

        manifest = pd.DataFrame({
            'probe': f['probe_metadata']['name'][:].astype(str),
            'lhs_probe': f['probe_metadata']['lhs_probe'][:].astype(str),
            'rhs_probe': f['probe_metadata']['rhs_probe'][:].astype(str),
            'gap_probe_sequence': f['probe_metadata']['gap_probe_sequence'][:].astype(str),
            'original_gap_probe_sequence': f['probe_metadata']['original_sequence'][:].astype(str),
        })
        if 'gene' in f['probe_metadata']:
            manifest['gene'] = f['probe_metadata']['gene'][:].astype(str)

        # Check if probe names are unique on the manifest
        if len(manifest['probe'].unique()) != len(manifest):
            raise ValueError("Probe names are not unique.")

        # Add reference to var_df
        var_df = var_df.merge(manifest, on='probe', how='left')
        var_df = var_df.rename(columns={'gap_probe_sequence': 'expected_gapfill', 'original_gap_probe_sequence': 'reference_gapfill'})
        var_df['probe_gapfill'] = var_df['probe'].str.cat(var_df['gapfill'], sep='|')
        var_df = var_df.set_index('probe_gapfill', drop=True)

        adata = ad.AnnData(X,
                           layers=layers,
                           obs=obs_df,
                           var=var_df,
                           uns={
                                "probe_metadata": manifest,
                                "plex": f.attrs['plex'],
                                "project": f.attrs['project'],
                                "created_date": f.attrs['created_date'], #pd.Timestamp(f.attrs['created_date']),
                                "n_cells": f.attrs['n_cells'],
                                "n_probes": f.attrs['n_probes'],
                                "n_probe_gapfill_combinations": f.attrs['n_probe_gapfill_combinations'],
                                "max_pcr_duplicates": f.attrs['max_pcr_duplicates'] if 'max_pcr_duplicates' in f.attrs else -1,
                           })

        if 'max_pcr_duplicates' in f.attrs and int(f.attrs['max_pcr_duplicates']) > 1:
            # We must read the pcr thresholds save the counts matrices for each threshold to the layers
            dup_grp = f['pcr_thresholded_counts']
            for threshold in range(1, f.attrs['max_pcr_duplicates']):
                adata.layers[f'X_pcr_threshold_{threshold}'] = read_sparse_matrix(dup_grp, f'pcr{threshold}')

    # Check if array_col and array_row exist in obs
    # If present, verify that all are integers
    if 'array_col' in adata.obs.columns and 'array_row' in adata.obs.columns:
        col_mask = adata.obs['array_col'].isnull() | (~np.issubdtype(adata.obs['array_col'].dtype, np.integer))
        row_mask = adata.obs['array_row'].isnull() | (~np.issubdtype(adata.obs['array_row'].dtype, np.integer))
        if col_mask.any() or row_mask.any():
            # We will need to regenerate only the problematic array_col and array_row values
            print("Warning: 'array_col' and 'array_row' in obs contain non-integer or null values. Regenerating problematic values.")
            # Create masks for problematic values
            problematic_mask = col_mask | row_mask

            if problematic_mask.any():
                # Vectorized parse from index -> base part before '-'
                idx_series = pd.Series(adata.obs.index.astype(str), index=adata.obs.index)
                base = idx_series.str.split('-', n=1).str[0]
                # Extract last two underscore-delimited tokens
                parts = base.str.rsplit('_', n=2, expand=True)
                if parts.shape[1] < 3:
                    parts = parts.reindex(columns=range(3))

                array_row_parsed = pd.to_numeric(parts.iloc[:, -2], errors='coerce').fillna(-1).astype(int)
                array_col_parsed = pd.to_numeric(parts.iloc[:, -1], errors='coerce').fillna(-1).astype(int)

                need_col = problematic_mask & col_mask
                need_row = problematic_mask & row_mask

                if need_col.any():
                    adata.obs.loc[need_col, 'array_col'] = array_col_parsed.loc[need_col].to_numpy()
                if need_row.any():
                    adata.obs.loc[need_row, 'array_row'] = array_row_parsed.loc[need_row].to_numpy()
            # Ensure columns are integer type
            adata.obs['array_col'] = adata.obs['array_col'].astype(int)
            adata.obs['array_row'] = adata.obs['array_row'].astype(int)

    return adata


# def merge_anndatas(adata_expression: ad.AnnData, adata_gapfill: ad.AnnData) -> ad.AnnData:
#     """
#     Merge two AnnData objects. The adata_gapfill should have the same barcodes as the adata_expression.
#     :param adata_expression: The expression data.
#     :param adata_gapfill: The gapfill data.
#     :return: The merged AnnData object.
#     """
#     # This will attempt to merge the two AnnData objects.
#     # Note that they have two completely different sets of vars so we will have to merge them manually by concatenating.
#
#     # First we will concatenate the expression data
#     X = scipy.sparse.hstack([adata_expression.X, adata_gapfill.X])
#     # For each layer, we will concatenate with empty matrices
#     layers = \
#         {k: scipy.sparse.hstack([v, scipy.sparse.csr_matrix((v.shape[0], adata_gapfill.X.shape[1]))]) for k, v in adata_expression.layers.items()} \
#         + {k: scipy.sparse.csr_matrix((adata_gapfill.X.shape[0], v.shape[1])) for k, v in adata_gapfill.layers.items()}
#     # Should be the same cells, so join the obs and fill in with NaNs for missing data
#     obs = pd.merge(adata_expression.obs, adata_gapfill.obs, how='outer', left_index=True, right_index=True)
#     # Concatenate the var data
#     # For each var, concatenate nan for filled in data
#     var = dict()
#     for column in adata_expression.var.columns:
#         var[column] = np.concatenate([adata_expression.var[column].values, np.full(adata_gapfill.X.shape[1], np.nan)])
#     for column in adata_gapfill.var.columns:
#         var[column] = np.concatenate([np.full(adata_expression.X.shape[1], np.nan), adata_gapfill.var[column].values])
#     var = pd.DataFrame(var)
#
#     uns = dict()
#     # Merge the uns data
#     for key in adata_expression.uns.keys():
#         uns[key] = adata_expression.uns[key]
#     for key in adata_gapfill.uns.keys():
#         uns[key] = adata_gapfill.uns[key]
#
#     # There may be varm or obsm data in the expression anndata, so we will have to merge them as well
#     varm = dict()
#     for key in adata_expression.varm.keys():
#         varm[key] = pd.concat([adata_expression.varm[key], pd.DataFrame(index=adata_gapfill.var.index)], axis=0)
#     for key in adata_gapfill.varm.keys():
#         varm[key] = pd.concat([pd.DataFrame(index=adata_expression.var.index), adata_gapfill.varm[key]], axis=0)
#
#     obsm = dict()
#     for key in adata_expression.obsm.keys():
#         obsm[key] = pd.concat([adata_expression.obsm[key], pd.DataFrame(index=adata_gapfill.obs.index)], axis=1)
#     for key in adata_gapfill.obsm.keys():
#         obsm[key] = pd.concat([pd.DataFrame(index=adata_expression.obs.index), adata_gapfill.obsm[key]], axis=1)
#
#     adata = ad.AnnData(X, layers=layers, obs=obs, var=var, uns=uns, varm=varm, obsm=obsm)
#     return adata


def compute_max_distance(seq_len: int, distance_per_10bp: int) -> int:
    """
    Computes the edit distance threshold given a sequence length.
    :param seq_len: The sequence length.
    :param distance_per_10bp: The distance per 10 bp.
    :return: The edit distance threshold. Minimum will be 1bp.
    """
    # Round up to the nearest 10bp
    return max(1, int(np.ceil(seq_len / 10) * distance_per_10bp))


def interpret_phred_letter(quality: str, base: Literal['sanger'] | Literal['illumina'] = 'illumina') -> float:
    """
    Convert a phred quality letter to a score.
    :param quality: The quality letter.
    :param base: The base quality system. Either 'sanger' or 'illumina'.
    :return: The probability of the base being incorrect.
    """
    assert len(quality) == 1, "Quality must be a single character."
    # Convert the character to a number
    score = ord(quality) - (33 if base == 'illumina' else 64)
    # Convert to P(error)
    return 10 ** (-score / 10)


def phred_string_to_probs(quality: str, base: Literal['sanger'] | Literal['illumina'] = 'illumina') -> list[float]:
    """
    Convert a phred quality string to a list of probabilities.
    :param quality: The quality string.
    :param base: The base quality system. Either 'sanger' or 'illumina'.
    :return: The list of probabilities.
    """
    return [interpret_phred_letter(q, base) for q in quality]


def permute_bases(seq: str, pos: list[int]) -> str:
    # Compute all possible sequences
    for combination in itertools.product("ACGT", repeat=len(pos)):
        curr_seq = seq
        for i, base in zip(pos, combination):
            curr_seq = curr_seq[:i] + base + curr_seq[i+1:]
        yield curr_seq


def generate_permuted_seqs(seq: str, quality: np.array, max_distance: int) -> str:
    # Generate all possible sequences with a maximum edit distance
    # We will prioritize the positions by the quality of the base
    # Sort by making the worst quality be first
    quality_indices = np.argsort(-quality)

    for base_positions in itertools.permutations(quality_indices, max_distance):
        yield from permute_bases(seq, base_positions)


# Based on: https://kb.10xgenomics.com/hc/en-us/articles/115003646912-How-is-sequencing-saturation-calculated
def sequencing_saturation(counts: np.array) -> float:
    """
    Sequencing saturation is 1 - (n_deduped_reads / n_reads)
    where n_deduped_reads is the number of valid cell bc/valid umi/gene combinations
    and n_reads is the total number of reads with a valid mapping to a valid cell barcode and umi.
    :param counts: Counts should be the number of reads rather than UMIs.
    :return: The saturation.
    """
    # Number of reads
    n_reads = counts.sum()
    # Number of deduped reads
    n_deduped_reads = (counts > 0).sum()
    return 1 - (n_deduped_reads / n_reads)


def sequence_saturation_curve(full_counts, n_points: int = 1_000) -> np.array:
    """
    Compute the sequencing saturation curve.
    :param full_counts: The cell x feature matrix where each count = # of reads..
    :param n_points: The number of points to compute the curve at. Note that this is computed on a log scale.
    :return: The saturation curve.
    """
    # Convert to dense
    if scipy.sparse.issparse(full_counts):
        full_counts = full_counts.toarray()
    full_counts = full_counts.astype(int)

    # Compute the subsampled proportion
    proportions = np.linspace(0.00001, 1, n_points)
    saturations = np.zeros((n_points,2))

    for i, proportion in enumerate(proportions):
        # Randomly subsample the data
        subsampled = np.random.binomial(n=full_counts, p=proportion, size=full_counts.shape)

        # Compute the saturation
        saturation = sequencing_saturation(subsampled)

        # Compute the mean reads/cell
        mean_reads_per_cell = subsampled.sum(axis=1).mean()

        saturations[i,0] = mean_reads_per_cell
        saturations[i,1] = saturation

    return saturations


def read_probes_input(probes: str) -> pd.DataFrame:
    if isinstance(probes, Path):
        probes = str(probes)
    # Parse the probes file
    if probes.endswith(".csv"):
        df = pd.read_csv(probes)
    elif probes.endswith(".xlsx"):
        df = pd.read_excel(probes)
    else:
        df = pd.read_table(probes)
    # Normalize the column names to be lowercase
    df.columns = df.columns.str.lower()
    if 'gap_probe_sequence' not in df.columns:
        if 'expected_gapfill' in df.columns:
            df.rename(columns={'expected_gapfill': 'gap_probe_sequence'}, inplace=True)
        else:
            df['gap_probe_sequence'] = "NA"
    if 'original_gap_probe_sequence' not in df.columns:
        if 'reference_gapfill' in df.columns:
            df.rename(columns={'reference_gapfill': 'original_gap_probe_sequence'}, inplace=True)
        else:
            df['original_gap_probe_sequence'] = "NA"
    gene_column = None
    # Check if there is a gene name column
    if 'gene' in df.columns:
        gene_column = 'gene'
    elif 'GENE' in df.columns:
        gene_column = 'GENE'
    elif 'symbol' in df.columns:
        gene_column = 'symbol'
    elif 'SYMBOL' in df.columns:
        gene_column = 'SYMBOL'
    elif 'gene_name' in df.columns:
        gene_column = 'gene_name'
    elif 'GENE_NAME' in df.columns:
        gene_column = 'GENE_NAME'
    elif 'gene_symbol' in df.columns:
        gene_column = 'gene_symbol'
    elif 'GENE_SYMBOL' in df.columns:
        gene_column = 'GENE_SYMBOL'
    # Rename the gene column to a standard name
    if gene_column is not None:
        df.rename(columns={gene_column: "gene"}, inplace=True)
    # Define the manifest with all data needed for downstream processing
    df = df[["name", "lhs_probe", "rhs_probe", "gap_probe_sequence", 'original_gap_probe_sequence'] + (
        [] if gene_column is None else ["gene"])]
    # Filter out non-unique entries
    df = df.drop_duplicates(subset=["lhs_probe", "rhs_probe", "gap_probe_sequence", 'original_gap_probe_sequence'] + (
        [] if gene_column is None else ["gene"]))
    # If there are duplicated names, add arbitrary suffixes
    if df.name.nunique() != df.shape[0]:
        print("Warning: Duplicated probe names found. Adding arbitrary suffixes to make them unique.")
        name_counts = df.name.value_counts()
        for name, count in name_counts.items():
            if count == 1:
                continue
            indices = df[df.name == name].index
            for i, idx in enumerate(indices):
                df.at[idx, "name"] = f"{name}_{i}"
    # Reset the index
    df.reset_index(drop=True, inplace=True)
    return df


def read_fastqs(read1s, read2s):
    r1_to_chain = []
    r2_to_chain = []
    for r1, r2 in zip(read1s, read2s):
        if r1.endswith(".gz"):
            read1_iterator = FastqGeneralIterator(gzip.open(r1, 'rt'))
        else:
            read1_iterator = FastqGeneralIterator(open(r1, 'r'))
        if r2.endswith(".gz"):
            read2_iterator = FastqGeneralIterator(gzip.open(r2, 'rt'))
        else:
            read2_iterator = FastqGeneralIterator(open(r2, 'r'))
        r1_to_chain.append(read1_iterator)
        r2_to_chain.append(read2_iterator)
    read1_iterator = itertools.chain(*r1_to_chain)
    read2_iterator = itertools.chain(*r2_to_chain)
    return read1_iterator, read2_iterator
