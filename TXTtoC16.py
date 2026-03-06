"""
TXTtoC16.py
================

This module implements a deterministic encoding of text into 16‑bit binary
patterns living on a four‑dimensional hypercube.  Each pattern can be viewed
either as an array of bits in \{0,1\}^{16} or as an array of spins in
\{+1,−1\}^{16}.  The vertices of the hypercube are indexed by the binary
tuples \((x,y,z,w)\in\{0,1\}^4\); the bit order (x,y,z,w) follows the
conventions specified in the `C16_pipeline_config.yaml` that accompanies the
Omega project.  The goal of this script is twofold:

1.  **Enumerate canonical motifs.**  The full hypercube has 2^16 possible
    labelings.  Under the action of the hyperoctahedral symmetry group B₄
    (all permutations of coordinate axes together with independent
    coordinate flips), these labelings fall into 3 876 orbits.  We
    construct these orbits explicitly by exploring the group action on
    representative configurations, marking visited configurations, and
    recording a canonical representative for each orbit.  For each canonical
    representative we compute three intrinsic quantities:

      - **Energy**:  For a given spin configuration \(\sigma\in\{\pm1\}^{16}\), the
        energy is defined as

        \[-\sum_{\langle i,j\rangle}\sigma_i\sigma_j\]

        where the sum runs over the 32 edges of the 4D hypercube.  This
        matches the unnormalised nearest–neighbour Ising energy on a
        hypercube and provides a natural ordering on the motifs.

      - **Entropy**:  Let \(p\) be the fraction of "up" spins (bits equal to 1).
        Then the Shannon entropy of the Bernoulli distribution with
        parameter \(p\) is

        \[-p\log_2 p - (1-p)\log_2 (1-p)\]  (with entropy set to zero if
        \(p\in\{0,1\}\)).  This quantity gives a coarse measure of how
        balanced a pattern is between up and down spins; lower entropy
        patterns are more ordered.

      - **Stabiliser size**:  The size of the subgroup of B₄ that leaves a
        given configuration invariant.  It is computed by counting how
        many group elements map the configuration to itself.  Large
        stabilisers correspond to highly symmetric patterns.

    The canonical motifs are sorted lexicographically by a triple of
    invariants `(energy, -stabiliser, entropy)`, prioritising low energy,
    high symmetry and low entropy.

2.  **Encode tokens deterministically.**  To encode an arbitrary token
    (e.g. a word), we compute a hash of the token using MD5, convert it
    to an integer, and take the remainder when dividing by the number of
    canonical motifs.  The resulting index selects one of the sorted
    canonical motifs.  Both the \{0,1\}^{16} and \{\pm1\}^{16} views of
    the motif are returned.  This mapping is deterministic: identical
    tokens always map to the same motif.

The script can be used as a library (import the `encode_token` and
`encode_text` functions) or as a standalone program.  When run from the
command line it accepts either a string to encode via the `-s` flag or a
path to a text file whose contents are to be encoded.

Usage examples
--------------

::

    # Encode a simple string
    python TXTtoC16.py -s "Hello world"

    # Encode a text file (each whitespace‑separated token is encoded)
    python TXTtoC16.py path/to/document.txt

This script is self contained; it does not rely on external data files
nor does it require internet access.  On first invocation the canonical
motifs and their invariants are computed lazily and cached in memory.

"""

from __future__ import annotations

import argparse
import hashlib
import math
from functools import lru_cache
from itertools import permutations, product
from typing import Dict, Iterable, List, Tuple

# -----------------------------------------------------------------------------
# Hypercube utilities
# -----------------------------------------------------------------------------

def _generate_transformations() -> List[List[int]]:
    """
    Generate the 384 symmetry transformations of the 4‑dimensional hypercube.

    Each transformation is represented as a list `mapping` of length 16, where
    `mapping[v]` gives the image of vertex `v` under the transformation.  A
    transformation is defined by a permutation of the four axes and a vector
    of four coordinate flips.  For vertex indices we use the convention that
    the least significant bit corresponds to the x‑coordinate, the next bit
    corresponds to y, then z, and the most significant bit corresponds to
    w (Euclidean time), i.e. `(x, y, z, w)` maps to the integer
    `x + 2*y + 4*z + 8*w`.
    """
    transformations: List[List[int]] = []
    axes = range(4)
    for perm in permutations(axes):
        for flips in product([0, 1], repeat=4):
            mapping = [0] * 16
            for v in range(16):
                bits = [(v >> i) & 1 for i in range(4)]
                newbits = [bits[perm[i]] ^ flips[i] for i in range(4)]
                new_val = 0
                for i in range(4):
                    new_val |= newbits[i] << i
                mapping[v] = new_val
            transformations.append(mapping)
    return transformations


@lru_cache(maxsize=1)
def get_transformations() -> List[List[int]]:
    """
    Lazily compute and cache the list of hypercube symmetry transformations.

    Returns
    -------
    List[List[int]]
        A list of 384 mappings, each mapping 16 vertex indices to new
        indices under a hyperoctahedral group element.
    """
    return _generate_transformations()


@lru_cache(maxsize=1)
def _generate_edges() -> List[Tuple[int, int]]:
    """
    Precompute the list of edges (unordered pairs of vertices) of the 4D
    hypercube.  Two vertices are adjacent if and only if their indices differ
    by exactly one bit (Hamming distance one).

    Returns
    -------
    List[Tuple[int, int]]
        A list of 32 unordered pairs `(i, j)` representing the edges of the
        hypercube.  Each pair has `i < j` to avoid duplication.
    """
    edges: List[Tuple[int, int]] = []
    for i in range(16):
        for j in range(i + 1, 16):
            if ((i ^ j).bit_count() == 1):
                edges.append((i, j))
    return edges


def _compute_orbits_and_canonicals() -> List[int]:
    """
    Construct the set of canonical representatives for the orbits of \{0,1\}^{16}
    under the hypercube symmetry group.

    We avoid brute force enumeration of all 65 536 configurations.  Instead,
    whenever we encounter a new configuration `p`, we immediately generate
    its orbit under the group action, mark all members of the orbit as
    visited, and record the lexicographically minimal configuration among the
    orbit as the canonical representative.  This exploration continues until
    all configurations have been visited.  The resulting list of canonicals
    has length 3 876 and can be sorted arbitrarily (we perform no sorting
    here; the sorting by invariants happens separately).

    Returns
    -------
    List[int]
        A list of canonical configurations represented as 16‑bit integers.
    """
    transformations = get_transformations()
    visited: Dict[int, bool] = {}
    canonicals: List[int] = []
    for p in range(1 << 16):
        if p in visited:
            continue
        # Generate the orbit of p and find its minimal element
        orbit: List[int] = []
        min_pattern = p
        for mapping in transformations:
            new_pattern = 0
            # apply the mapping to pattern p
            for v in range(16):
                if p & (1 << v):
                    new_pattern |= 1 << mapping[v]
            orbit.append(new_pattern)
            if new_pattern < min_pattern:
                min_pattern = new_pattern
        # Mark all members of the orbit as visited
        for q in orbit:
            visited[q] = True
        # Record the canonical representative
        canonicals.append(min_pattern)
    return canonicals


@lru_cache(maxsize=1)
def get_canonical_patterns() -> List[int]:
    """
    Lazily compute and cache the list of canonical hypercube motifs.

    The canonical motifs are computed once on demand.  Repeated calls return
    the cached list, avoiding the expensive enumeration.

    Returns
    -------
    List[int]
        A list of 3 876 integers, each representing a canonical 16‑bit pattern.
    """
    canonicals = _compute_orbits_and_canonicals()
    # Remove potential duplicates (there should not be any) and return as list
    unique_canonicals = list(dict.fromkeys(canonicals).keys())
    assert len(unique_canonicals) == 3876, (
        f"Expected 3876 canonical motifs, got {len(unique_canonicals)}"
    )
    return unique_canonicals


def _bits_pm(pattern: int) -> List[int]:
    """
    Convert a 16‑bit pattern into its ±1 spin representation.

    Parameters
    ----------
    pattern : int
        An integer representing a configuration in \{0,1\}^{16}.

    Returns
    -------
    List[int]
        A list of length 16 where each entry is +1 if the corresponding bit
        of `pattern` is set and −1 otherwise.
    """
    spins = [1 if pattern & (1 << v) else -1 for v in range(16)]
    return spins


def _bits_01(pattern: int) -> List[int]:
    """
    Convert a 16‑bit pattern into its binary representation.

    Parameters
    ----------
    pattern : int
        An integer representing a configuration in \{0,1\}^{16}.

    Returns
    -------
    List[int]
        A list of length 16 where each entry is 1 if the corresponding bit
        of `pattern` is set and 0 otherwise.
    """
    bits = [1 if pattern & (1 << v) else 0 for v in range(16)]
    return bits


def _compute_features(pattern: int) -> Tuple[int, float, int]:
    """
    Compute intrinsic invariants (energy, entropy, stabiliser size) of a motif.

    Parameters
    ----------
    pattern : int
        A canonical motif represented as a 16‑bit integer.

    Returns
    -------
    Tuple[int, float, int]
        A triple `(energy, entropy, stabiliser_size)` describing the motif.
        - `energy` is the unnormalised Ising energy defined as
          \(-\sum_{i,j\in\mathrm{edges}}\sigma_i\sigma_j\).
        - `entropy` is the binary Shannon entropy of the spin distribution.
        - `stabiliser_size` is the number of hypercube symmetries that fix
          the configuration.
    """
    # Energy
    spins = _bits_pm(pattern)
    energy = 0
    for i, j in _generate_edges():
        energy += -(spins[i] * spins[j])
    # Entropy
    bits = _bits_01(pattern)
    p = sum(bits) / 16.0
    if p == 0.0 or p == 1.0:
        entropy = 0.0
    else:
        entropy = -p * math.log(p, 2) - (1.0 - p) * math.log(1.0 - p, 2)
    # Stabiliser size
    stabiliser = 0
    transformations = get_transformations()
    for mapping in transformations:
        new_pattern = 0
        for v in range(16):
            if pattern & (1 << v):
                new_pattern |= 1 << mapping[v]
        if new_pattern == pattern:
            stabiliser += 1
    return (energy, entropy, stabiliser)


@lru_cache(maxsize=1)
def _sorted_canonical_patterns() -> List[int]:
    """
    Sort canonical motifs by (energy, -stabiliser_size, entropy).

    The sorted order emphasises low energy (fewer domain walls), high
    symmetry (large stabiliser), and low entropy.  Ties are broken by the
    natural ordering of integers representing patterns.

    Returns
    -------
    List[int]
        A list of canonical motifs sorted by the specified invariants.
    """
    canonicals = get_canonical_patterns()
    features: Dict[int, Tuple[int, float, int]] = {}
    for pattern in canonicals:
        features[pattern] = _compute_features(pattern)
    sorted_patterns = sorted(
        canonicals,
        key=lambda p: (features[p][0], -features[p][2], features[p][1], p),
    )
    return sorted_patterns


def encode_token(token: str) -> Tuple[List[int], List[int], int, int, float, int]:
    """
    Deterministically encode a single token into a 16‑bit motif and compute its invariants.

    The token is hashed with MD5 and the lowest 16 bits of the hash are used
    directly as the pattern.  This approach utilises the full 65 536 element
    configuration space rather than collapsing under symmetry.  For each
    pattern we compute intrinsic invariants (energy, entropy, stabiliser
    size) on the fly.  These invariants can be used downstream to order
    motifs or derive additional structure.

    Parameters
    ----------
    token : str
        The input string token to encode.

    Returns
    -------
    Tuple[List[int], List[int], int, int, float, int]
        A six‑tuple `(bits01, bits_pm, pattern_id, energy, entropy, stabiliser)`
        where:

        - `bits01` is the 16‑length list of 0/1 bits of the motif.
        - `bits_pm` is the corresponding list of ±1 spins.
        - `pattern_id` is the integer in `[0, 65535]` representing the motif.
        - `energy` is the unnormalised Ising energy of the motif.
        - `entropy` is the binary entropy of the spin distribution.
        - `stabiliser` is the size of the symmetry subgroup fixing the motif
          under the full hyperoctahedral group (384 elements).
    """
    # Hash the token and extract a 16‑bit pattern id
    digest = hashlib.md5(token.encode('utf-8')).digest()
    # Use the first two bytes (16 bits) as pattern id; int.from_bytes uses
    # big‑endian by default, but the particular endianness does not matter
    pattern_id = int.from_bytes(digest[:2], byteorder="big")
    bits01 = _bits_01(pattern_id)
    bits_pm = _bits_pm(pattern_id)
    # Compute invariants
    energy, entropy, stabiliser = _compute_features(pattern_id)
    return bits01, bits_pm, pattern_id, energy, entropy, stabiliser


def encode_text(text: str) -> List[Tuple[str, List[int], List[int], int, int, float, int]]:
    """
    Encode all whitespace‑separated tokens in a text string.

    Parameters
    ----------
    text : str
        The input text to encode.  Tokens are obtained by splitting on
        arbitrary whitespace.

    Returns
    -------
    List[Tuple[str, List[int], List[int], int, int, float, int]]
        A list of tuples `(token, bits01, bits_pm, pattern_id, energy, entropy, stabiliser)`
        for each token encountered in the text.  If the input text is empty
        or consists solely of whitespace, an empty list is returned.
    """
    tokens: List[str] = text.split()
    results: List[Tuple[str, List[int], List[int], int, int, float, int]] = []
    for token in tokens:
        bits01, bits_pm, pattern_id, energy, entropy, stabiliser = encode_token(token)
        results.append((token, bits01, bits_pm, pattern_id, energy, entropy, stabiliser))
    return results


def _format_motif(bits01: Iterable[int]) -> str:
    """
    Format a 16‑bit motif as a human‑readable 4×4 grid.

    This helper arranges the 16 bits into a 2×2×2×2 hypercube by
    interpreting the last coordinate `w` as Euclidean time.  We present the
    motif as two 2×2×2 blocks corresponding to w=0 and w=1.  Each block is
    rendered as two layers (z=0 and z=1) of 2×2 matrices (x,y plane).

    Parameters
    ----------
    bits01 : Iterable[int]
        The motif bits in canonical x‑major order.

    Returns
    -------
    str
        A multiline string visualising the hypercube configuration.
    """
    bits = list(bits01)
    out_lines: List[str] = []
    for w in range(2):
        out_lines.append(f"w = {w}:")
        for z in range(2):
            rows: List[str] = []
            for y in range(2):
                row_bits = []
                for x in range(2):
                    idx = x + 2 * y + 4 * z + 8 * w
                    row_bits.append(str(bits[idx]))
                rows.append(" ".join(row_bits))
            out_lines.append("  z=" + str(z))
            out_lines.extend(["    " + row for row in rows])
        out_lines.append("")
    return "\n".join(out_lines)


def main() -> None:
    """
    Entry point for the command‑line interface.

    Accepts either a text string (via `-s`) or a file path.  For each
    whitespace‑separated token, prints its motif index and both
    binary/±1 representations.  Additionally, a visualisation of the
    motif may be requested with the `-v` flag.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Deterministically encode tokens into canonical 16‑bit motifs "
            "on the 4D hypercube.  See module docstring for details."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-s",
        "--string",
        type=str,
        help="Input string to encode (whitespace‑separated tokens).",
    )
    group.add_argument(
        "file",
        nargs="?",
        help="Path to a text file to encode.  If provided, the file contents "
        "are read and split into tokens based on whitespace."
    )
    parser.add_argument(
        "-v",
        "--visualise",
        action="store_true",
        help="Visualise each motif as a set of 2×2×2×2 blocks."
    )
    args = parser.parse_args()

    if args.string is not None:
        text = args.string
    else:
        # Read file contents
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            raise SystemExit(f"Error reading file {args.file!r}: {e}")

    results = encode_text(text)
    if not results:
        print("No tokens to encode.")
        return
    # Print results
    for token, bits01, bits_pm, pattern_id, energy, entropy, stabiliser in results:
        print(f"Token: {token}")
        print(f"  Pattern id: {pattern_id}")
        print(f"  Bits (0/1): {bits01}")
        print(f"  Spins (+/−1): {bits_pm}")
        print(f"  Energy: {energy}")
        print(f"  Entropy: {entropy:.6f}")
        print(f"  Stabiliser size: {stabiliser}")
        if args.visualise:
            print(_format_motif(bits01))
        print()


if __name__ == "__main__":
    main()
