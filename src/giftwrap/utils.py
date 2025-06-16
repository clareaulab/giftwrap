import functools
import gzip
import io
import json
import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from tempfile import NamedTemporaryFile
import itertools
import contextlib
from typing import Literal, Optional, Union
import multiprocessing

import math
import numpy as np
import pandas as pd
import h5py
import anndata as ad
import scipy
import diskcache as dc
from Bio.SeqIO.QualityIO import FastqGeneralIterator


class PrefixTree:
    """
    Dynamically expanding tree (up to N mismatches) of strings based on prefixes.
    Note that indels are not considered since it would be extremely slow to search.
    This is based on the trie data structure.
    Note that this supports memory sharing.
    """

    __slots__ = ['_root', '_size', 'allow_indels', '_cache', '_prefix_only_search']

    def __init__(self,
                 contents: list[str],
                 allow_indels: bool = False,
                 prefix_only_search: bool = True):
        """
        Initialize the tree with a list of contents.
        :param contents: The list of contents.
        :param allow_indels: Whether to allow insertions/deletions in the search.
            Equivalent to edit distance when true, equivalent to hamming distance when false.
        :param prefix_only_search: If true, the tree will only allow for insertions at the start of the string. And will attempt to only match the longest prefix in the query.
        """
        self.allow_indels = allow_indels
        self._root = dict()
        self._size = 0
        self._prefix_only_search = prefix_only_search

        # initialize cache for lookups
        self._cache = dc.Cache()

        for content in contents:
            self.add(content)

    @property
    def root(self) -> dict:
        return self._root

    def add(self, content: str):
        """
        Add a content to the tree.
        """
        node = self.root
        for char in content:
            if char not in node:
                node[char] = dict()
            node = node[char]
        if '$' not in node:  # This end node did not exist yet
            node['$'] = content  # Store the full content at the end as a short cut
            self._size += 1

    # Recursive function to find the best match as fast as possible
    # Note that we do not allow for insertions and deletions, only substitutions
    def _search(self,
                full_content: str,
                curr_node: dict,
                content_index: int,
                max_mismatches: int,
                curr_mismatches: int,
                visited: dict,
                curr_depth: int = 0,
                start_index: int = 0) -> Optional[tuple[str, int, int]]:
        if id(curr_node) in visited:  # Already visited this node
            if curr_mismatches >= visited[id(curr_node)]:  # Skip if we have already visited this node with a better score than is possible
                return None
        # Check for end cases
        if curr_mismatches > max_mismatches:
            visited[id(curr_node)] = max_mismatches+1
            return None
        if content_index > len(full_content):
            visited[id(curr_node)] = max_mismatches+1  # Mark this node as visited
            return None
        remaining_mismatches = max_mismatches - curr_mismatches

        best_score = None
        best = None

        if content_index == len(full_content):  # We reached the end of the content, is there an end of word here?
            if '$' in curr_node:  # This will always be the best match
                # Add to visited
                self._insert_if_better(visited, id(curr_node), curr_mismatches)
                return curr_node['$'], curr_mismatches, start_index
            elif not (self.allow_indels and remaining_mismatches <= 0):  # Stop early if we can't insert/delete
                visited[id(curr_node)] = max_mismatches+1  # Mark this node as visited
                return None  # We have reached the end of the content and have no more mismatches to spare

        # See if a match exists or perform mismatches if this is not the end of the content
        if content_index < len(full_content):
            if full_content[content_index] in curr_node:
                # Continue the search
                res = self._search(full_content, curr_node[full_content[content_index]], content_index + 1, max_mismatches, curr_mismatches, visited, curr_depth + 1, start_index)
                if res is not None:
                    if best_score is None or res[1] < best_score:
                        best = res
                        best_score = res[1]
                    self._insert_if_better(visited, id(curr_node), res[1])
                    return best

            # No match, try mismatches
            for char in curr_node:
                if char == full_content[content_index]:  # Ignore already scanned characters
                    continue
                elif char == '$':  # End of word, but not end of the query. Continue to see if there are other matches
                    continue
                # Continue the search, but with a mismatch
                res = self._search(full_content, curr_node[char], content_index + 1, max_mismatches, curr_mismatches + 1, visited, curr_depth + 1, start_index)
                if res is not None:
                    if best is None or res[1] < best_score:
                        best = res
                        best_score = res[1]
                        # Break early if we have a perfect match with this substitution
                        if best_score == curr_mismatches+1:
                            break

        if best_score is None or best_score > max_mismatches or (self.allow_indels and (best_score - curr_mismatches) > 1):
            # No match found, or invalid match found, or the match that was found had more than one mismatch.
            # Meaning we can theoretically improve the score with indels
            if not self.allow_indels:
                visited[id(curr_node)] = max_mismatches+1
                return None  # No need to continue if we don't allow indels

            # If we are prefix-matching, the query string may be too long
            if self.allow_indels and self._prefix_only_search:
                if '$' in curr_node and curr_mismatches <= max_mismatches:
                    # By this point, we should have already found a match if it existed
                    self._insert_if_better(visited, id(curr_node), curr_mismatches)
                    return curr_node['$'], curr_mismatches, start_index
                elif curr_depth == 0:  # This is the top-level door, try removing the current position without penalty, but reduce the number of further mismatches
                    res = self._search(full_content[1:], curr_node, content_index, max_mismatches-1, curr_mismatches, visited, curr_depth, start_index + 1)
                    if res is not None:
                        if best is None or res[1] < best_score:
                            best = res
                            best_score = res[1]
                        # Break early if we have a perfect match with this deletion
                        if best_score == curr_mismatches:
                            self._insert_if_better(visited, id(curr_node), best_score)
                            return best
            else:
                # First test deleting this position (i.e. the string has an insertion)
                res = self._search(full_content, curr_node, content_index + 1, max_mismatches, curr_mismatches + 1, visited, curr_depth + 1, start_index)  # Not updating start index since we are considering this an error
                if res is not None:
                    if best is None or res[1] < best_score:
                        best = res
                        best_score = res[1]
                    # Break early if we have a perfect match with this deletion
                    if best_score == curr_mismatches+1:
                        self._insert_if_better(visited, id(curr_node), best_score)
                        return best

                # Now test inserting a character
                for char in curr_node:
                    if char == '$':
                        continue
                    res = self._search(full_content, curr_node[char], content_index, max_mismatches, curr_mismatches + 1, visited, curr_depth + 1, start_index)
                    if res is not None:
                        if best is None or res[1] < best_score:
                            best = res
                            best_score = res[1]
                        # Break early if we have a perfect match with this insertion
                        if best_score == curr_mismatches+1:
                            self._insert_if_better(visited, id(curr_node), best_score)
                            return best

        self._insert_if_better(visited, id(curr_node), best_score)

        return best

    def search(self, content: str, max_mismatches: int) -> Optional[tuple[str, int, int]]:
        """
        Search the tree for a content with a maximum number of mismatches.

        Returns None if no content is found with the given number of mismatches.
        Else returns a tuple of: Corrected Content, Number of Mismatches, Start Index within the Given String the match begins on
        """
        cached = self._cache.get((content, max_mismatches), default=-1)
        if cached != -1:
            return cached

        res = self._search(content, self.root, 0, max_mismatches, 0, dict())

        self._cache[(content, max_mismatches)] = res

        if res is not None and res[1] <= max_mismatches:
            return res
        return None

    def _insert_if_better(self, dictionary, key, value):
        if value is None:
            return False

        if key not in dictionary:
            dictionary[key] = value
            return True
        elif value < dictionary[key]:
            dictionary[key] = value
            return True
        return False

    def __contains__(self, content: str) -> bool:
        """
        Check if exact content is in the tree.
        """
        node = self.root
        for char in content:
            if char not in node:
                return False
            node = node[char]
        return '$' in node

    def values(self) -> list[str]:
        """
        Return all values in the tree.
        """
        vals = []
        self._values(self.root, vals)
        return vals

    def _values(self, node: dict, vals: list[str]):
        """
        Recursively extract all values from the tree.
        """
        if '$' in node:
            vals.append(node['$'])
        for char in node:
            if char == '$':
                continue
            self._values(node[char], vals)

    def size(self) -> int:
        return self._size

    def __len__(self):
        return self.size()


class SequentialPrefixTree(PrefixTree):
    """
    A prefix tree that can be searched sequentially. I.e. for matching pairs of barcodes.
    """

    def __init__(self, trees: list[PrefixTree]):
        """
        Initialize a prefix tree which sequentially matches strings.
        :param trees: The trees to use.
        """
        assert len(trees) > 1, "Must have at least two trees."
        assert all([tree._prefix_only_search for tree in trees]), "All trees must be prefix only search."
        self._trees = trees
        super().__init__([], allow_indels=trees[0].allow_indels, prefix_only_search=trees[0]._prefix_only_search)
        self._size = math.prod(map(len, trees))

    def values(self) -> list[str]:
        """
        Return all values in the tree.
        """
        return list(itertools.product(*[tree.values() for tree in self._trees]))

    def search(self, content: str, max_mismatches: int) -> Optional[tuple[str, int, int]]:
        curr_content = content
        corrected_content = ""
        remaining_mismatches = max_mismatches
        first_start = None
        for tree in self._trees:
            if len(curr_content) == 0 or remaining_mismatches < 0:
                return None
            res = tree.search(curr_content, remaining_mismatches)
            if res is None:
                return None
            corrected_content += res[0]
            remaining_mismatches -= res[1]
            curr_content = curr_content[res[2] + len(res[0]):]
            if first_start is None:
                first_start = res[2]

        return corrected_content, max_mismatches - remaining_mismatches, first_start

    # These functions don't make sense here:

    @property
    def root(self):
        raise NotImplementedError()

    def add(self, content: str):
        raise NotImplementedError()

    def __contains__(self, content: str) -> bool:
        raise NotImplementedError()  # TODO


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

    def __init__(self, barcode_dir: Optional[str] = None, read1_length: Optional[int] = None, read2_length: Optional[int] = None):
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
    def barcode_tree(self) -> PrefixTree:
        """
        Return a prefix tree (trie) of the cell barcodes for fast mismatch searches.
        :return: The tree.
        """
        return PrefixTree(self.cell_barcodes)

    @functools.lru_cache(1024)
    def correct_barcode(self, read: str, max_mismatches: int, start_idx: int, end_idx: int) -> tuple[Optional[str], bool]:
        """
        Given a probable barcode string, attempt to correct the sequence.
        :param read: The barcode-containing sequence.
        :param max_mismatches: The maximum number of mismatches to allow.
        :param start_idx: The start index of the barcode in the read.
        :param end_idx: The end index of the barcode in the read.
        :return: The corrected barcode, or None if no match was found.
        """
        res = self.barcode_tree.search(read[start_idx:end_idx], max_mismatches)
        if res is not None:
            return res[0], res[1] > 0
        return None, False


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
        self._barcodes = PrefixTree(list(barcodes.unique()))

        self._probe_barcodes = _tx_barcode_oligos

        self._index_to_probe_barcodes = _tx_barcode_to_oligo


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
    def barcode_tree(self) -> PrefixTree:
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
        self._barcodes = PrefixTree(barcodes["barcode"].tolist())
        self._barcode_coordinates = {row["barcode"]: (row["x"], row["y"]) for _, row in barcodes.iterrows()}

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
    def barcode_tree(self) -> PrefixTree:
        return self._barcodes


class VisiumHDFormatInfo(TechnologyFormatInfo):

    def __init__(self,
                 space_ranger_path: Optional[str] = None,
                 barcode_dir: Optional[str] = None,
                 read1_length: Optional[int] = 43,
                 read2_length: Optional[int] = 50,
                 barcode_list: Optional[list[Path]] = None):
        if barcode_dir is None and barcode_list is None:
            raise ValueError("Either barcode_dir or barcode_list must be provided.")

        super().__init__(barcode_dir, read1_length, read2_length)
        xy_whitelist = None
        if barcode_list:
            barcodes = _parse_possible_barcodes(barcode_list)
            if barcodes:
                xy_whitelist = set()
                # Parse the coordinates from the barcode strings
                for bc in barcodes:
                    y, x = bc.split("-")[0].split("_")[1:]
                    y = int(y)
                    x = int(x)
                    xy_whitelist.add((x, y))

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
        self._bc1_tree = PrefixTree(slide_def.two_part.bc1_oligos)
        self._bc2_tree = PrefixTree(slide_def.two_part.bc2_oligos)
        self._bc_lengths = set()
        # _barcode_tree = PrefixTree([], allow_indels=True)  # Allow for indels as per 10X:
        # https://www.10xgenomics.com/support/software/space-ranger/latest/algorithms-overview/gene-expression#:~:text=using%20the%20edit%20distance%2C%20which%20allows%20for%20insertions%2C%20deletions%2C%20and%20substitutions.%20Up%20to%20four%20edits%20are%20permissible%20to%20correct%20a%20barcode%20to%20the%20whitelist.
        for x, bc1 in enumerate(slide_def.two_part.bc1_oligos):
            for y, bc2 in enumerate(slide_def.two_part.bc2_oligos):
                if (x, y) not in xy_whitelist:
                    continue   # Skip if not in the whitelist
                self._bc_lengths.add(len(bc1))
                self._bc_lengths.add(len(bc2))
                cell_bc = bc1 + bc2
                self._barcode_coordinates[cell_bc] = (x, y)
                # _barcode_tree.add(cell_bc)
        # self._barcode_tree = SequentialPrefixTree([bc1_tree, bc2_tree])

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
    def barcode_tree(self) -> PrefixTree:
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
        search_lengths.sort()
        return search_lengths

    @property
    @functools.lru_cache(1)
    def _default_range(self):
        return ((self._segment1_offset, self._segment1_offset + self._segment1_length),
         (self._segment2_offset, self._segment2_offset + self._segment2_length))

    @functools.lru_cache(1)
    def possible_pairing_ranges(self, read_len: int) -> list[tuple[tuple[int, int],tuple[int, int]]]:
        possible_pairing_ranges = []
        for offset in range(self._min_offset, self._max_offset+1):
            # For each offset, try every possible combination of the two barcodes
            for bc1_len, bc2_len in itertools.product(self._bc_lengths, self._bc_lengths):
                # Skip if the length would go off the read
                if bc1_len + bc2_len + offset > read_len:
                    continue

                possible_pairing_ranges.append(((offset, offset+bc1_len), (offset+bc1_len, offset+bc1_len+bc2_len)))

        # Add the default range from the chemistry def . We fall back to this if there are no best matches
        possible_pairing_ranges.append(self._default_range)

        return possible_pairing_ranges

    # Override the search to search all trees
    @functools.lru_cache(1024)
    def correct_barcode(self, read: str, max_mismatches: int, start_idx: int, end_idx: int) -> tuple[Optional[str], bool]:
        # Based on logic from spaceranger: https://github.com/10XGenomics/spaceranger/blob/main/lib/rust/cr_lib/src/stages/barcode_correction.rs#L100-L235
        # and https://github.com/10XGenomics/spaceranger/blob/main/lib/rust/cr_types/src/rna_read.rs#L297-L349
        # Note that we ignore the provided start and end indices, since we will follow 10X's logic exactly

        # Additionally note that to match spaceranger we will allow for 4 mismatches
        max_mismatches = max(max_mismatches, 4)

        possible_pairing_ranges = self.possible_pairing_ranges(len(read))

        # Now we iterate and try to find exact matches
        matches_i = []
        for pairing_range in possible_pairing_ranges:
            # Get the ranges
            bc1_range, bc2_range = pairing_range
            # Get the barcodes
            bc1 = read[bc1_range[0]:bc1_range[1]]
            bc2 = read[bc2_range[0]:bc2_range[1]]
            # Check if they are in the trees
            if bc1 in self._bc1_tree and bc2 in self._bc2_tree:
                matches_i.append((bc1, bc2, bc1_range, bc2_range))

        if len(matches_i) == 0:
            pass  # No exact matches found
        elif len(matches_i) == 1:  # Only one match found
            bc1, bc2, bc1_range, bc2_range = matches_i[0]
            return bc1 + bc2, False  # No mismatches
        else:  # Multiple matches found, pick the largest match (or default)
            bc1, bc2, bc1_range, bc2_range = matches_i[-1]
            return bc1 + bc2, False  # No mismatches

        # Now we see if either bc1 or bc2 has an exact match for the default range
        default_bc1 = read[self._default_range[0][0]:self._default_range[0][1]]
        default_bc2 = read[self._default_range[1][0]:self._default_range[1][1]]

        search_lengths = self.get_lengths_to_search()

        if default_bc1 in self._bc1_tree:  # Default BC1 is found!
            # Starting from the end of bc1, we will search for the bc2
            bc1_end = self._default_range[0][1]
            possible_bc2s = []
            for length in search_lengths:
                if bc1_end + length > len(read):
                    continue
                possible_bc2 = read[bc1_end:bc1_end + length]
                # Search for a valid sequence
                res = self._bc2_tree.search(possible_bc2, max_mismatches)
                if res is not None:
                    corrected = default_bc1 + res[0]
                    distance = res[1]
                    if distance <= 1:  # Can't theoretically have more than 1 mismatch as the best match, so exit early
                        return corrected, distance > 0
                    possible_bc2s.append((corrected, distance))
            # Find the lowest score
            if len(possible_bc2s) > 0:
                best = min(possible_bc2s, key=lambda x: x[1])
                return best[0], best[1] > 0
        elif default_bc2 in self._bc2_tree:  # Default BC2 is found!
            # Same as above, but we gotta search for bc1
            bc2_start = self._default_range[1][0]
            possible_bc1s = []
            for length in search_lengths:
                if bc2_start - length < 0:
                    continue
                possible_bc1 = read[bc2_start - length:bc2_start]
                # Search for a valid sequence
                res = self._bc1_tree.search(possible_bc1, max_mismatches)
                if res is not None:
                    corrected = res[0] + default_bc2
                    distance = res[1]
                    if distance <= 1:
                        return corrected, distance > 0
                    possible_bc1s.append((corrected, distance))
            # Find the lowest score
            if len(possible_bc1s) > 0:
                best = min(possible_bc1s, key=lambda x: x[1])
                return best[0], best[1] > 0
        else:  # Nothing found...
            # We have to fuzzy match bc1 and bc2
            # We will search for the longest possible match
            best_distance = 1e6
            best_bc = None
            for possible_bc1_start in range(self._min_offset, self._max_offset+1):
                for bc1_len, bc2_len in itertools.product(search_lengths, search_lengths):
                    if possible_bc1_start + bc1_len + bc2_len > len(read):
                        continue
                    possible_bc1 = read[possible_bc1_start:possible_bc1_start + bc1_len]
                    possible_bc2 = read[possible_bc1_start + bc1_len:possible_bc1_start + bc1_len + bc2_len]
                    # Search for a valid sequence
                    bc1_res = self._bc1_tree.search(possible_bc1, max_mismatches)
                    if bc1_res is None:
                        continue
                    bc1_corrected = bc1_res[0]
                    bc1_distance = bc1_res[1]
                    bc2_res = self._bc2_tree.search(possible_bc2, max_mismatches)
                    if bc2_res is None:
                        continue
                    bc2_corrected = bc2_res[0]
                    bc2_distance = bc2_res[1]
                    if bc1_distance + bc2_distance > max_mismatches:
                        continue
                    if bc1_distance + bc2_distance < best_distance:
                        best_distance = bc1_distance + bc2_distance
                        best_bc = bc1_corrected + bc2_corrected

                        if best_distance <= 1:  # Literally can't get better than this so exit early
                            return best_bc, best_distance > 0

            if best_bc is not None:
                return best_bc, best_distance > 0

        # No matches found, return None
        return None, False


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
        output_file.write(f"cell_barcode\tlhs_probe\trhs_probe\tcalled_probe\tgapfill\tpcr_duplicates\tpercent_supporting\n")
        for line in input_file:
            split = line.strip().split("\t")
            if len(split) != 6:
                continue
            cell_idx, probe_idx, probe_bc_idx, gapfill, umi_count, percent_supporting = line.strip().split("\t")
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

            output_file.write(f"{cell_barcode}\t{lhs_probe}\t{rhs_probe}\t{probe_bc_idx}\t{gapfill}\t{umi_count}\t{percent_supporting}\n")


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


def filter_h5_file(input_file: Path, output_file: Path, barcodes_list: np.ndarray[str], pad_matrix: bool = True):
    """
    Given a counts h5 file and a list of barcodes, filter the barcodes to only include the ones in the list.
    :param input_file: The input h5 file.
    :param output_file: The output h5 file.
    :param barcodes_list: The barcodes list.
    :param pad_matrix: Whether to pad the matrix with zeros if there are barcodes provided that don't exist in the file.
    """
    # First, copy the file
    shutil.copy(input_file, output_file)
    # Then open the file
    with h5py.File(output_file, 'r+') as f:
        barcodes = f['matrix']['barcode'][:].astype(str)
        # Get the filtered list of barcodes indices and re-order them
        barcode_indices = np.array([i for i, bc in enumerate(barcodes) if bc in barcodes_list])
        # Check if the we need to filter the data
        if len(barcode_indices) == len(barcodes):
            return  # Equal size, no point in filtering

        if len(barcode_indices) == 0:
            raise ValueError("No barcodes found in the file.")

        if pad_matrix and len(barcodes_list) > len(barcode_indices):
            padded_barcodes = [bc for bc in barcodes_list if bc not in barcodes]
            print(f"Padding {len(padded_barcodes)} unseen cells with zeroes.")
        else:
            padded_barcodes = []

        # Filter the data
        del f['matrix']['barcode']
        f['matrix'].create_dataset("barcode",
                                  data=np.concatenate([barcodes[barcode_indices].astype('S'), np.array(padded_barcodes, dtype='S')]),
                                  compression='gzip')

        for layer_name in ['data', 'total_reads', 'percent_supporting']:
            data = read_sparse_matrix(f['matrix'], layer_name)
            data = data[barcode_indices, :]
            # Add padding
            if len(padded_barcodes) > 0:
                data = scipy.sparse.vstack([data, scipy.sparse.csr_matrix((len(padded_barcodes), data.shape[1]))])
            del f['matrix'][layer_name]
            write_sparse_matrix(f['matrix'], layer_name, data)

        obs_meta_columns = f['cell_metadata']['columns'][:].astype(str)
        obs_meta_df = pd.DataFrame({
            col: (f['cell_metadata'][col][:].astype(str) if col == 'barcode' else f['cell_metadata'][col][:].astype(int)) for col in obs_meta_columns
        }).set_index('barcode')
        obs_meta_df = obs_meta_df.loc[barcodes[barcode_indices]]
        # Move the index back to a column
        obs_meta_df = obs_meta_df.reset_index()
        # Add padding
        if len(padded_barcodes) > 0:
            obs_meta_df = pd.concat([obs_meta_df, pd.DataFrame({col: (([pd.NA] * len(padded_barcodes)) if col != "barcode" else padded_barcodes) for col in obs_meta_df.columns})])
        del f['cell_metadata']
        cell_metadata_grp = f.create_group('cell_metadata')
        cell_metadata_grp.create_dataset('columns', data=np.array(obs_meta_df.columns, dtype='S'), compression='gzip')
        for col in obs_meta_df.columns:
            values = obs_meta_df[col].values
            # If it is not an integer or float, then convert to string
            if not np.issubdtype(values.dtype, np.number):
                values = values.astype('S')
            cell_metadata_grp.create_dataset(col, data=values, compression='gzip')

        del f.attrs['n_cells']
        f.attrs['n_cells'] = len(barcodes_list) + len(padded_barcodes)

        # Done


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
        obs_df = pd.DataFrame({
            'barcode': f['matrix']['barcode'][:].astype(str),
        }).set_index('barcode')

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
            'original_gap_probe_sequence': f['probe_metadata']['gap_probe_sequence'][:].astype(str),
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
                           })

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
        df['gap_probe_sequence'] = "NA"
    if 'original_gap_probe_sequence' not in df.columns:
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
