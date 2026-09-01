import re
from collections import defaultdict

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from scipy.interpolate import interp1d


_MOLECULAR_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)([1-9]\d*)?", re.ASCII)
_ELEMENT_SYMBOLS = frozenset(
    Chem.GetPeriodicTable().GetElementSymbol(atomic_number)
    for atomic_number in range(1, 119)
)
_INVALID_FORMULA_SYNTAX = (
    "Cannot parse the molecular formula because its complete text must contain "
    "only element symbols and optional positive atom counts."
)


def get_atom_counts_from_formula(formula_string: str) -> dict[str, int]:
    """Parse one complete elemental formula into positive atom counts.

    The whole string must consist of recognized element symbols followed by
    optional positive decimal counts. Rejecting partial matches keeps unrelated
    prose and malformed counts from silently changing candidate filtering.
    """

    if not isinstance(formula_string, str):
        raise TypeError("Cannot parse the molecular formula because it is not text.")

    counts = defaultdict(int)
    position = 0
    while position < len(formula_string):
        token = _MOLECULAR_FORMULA_TOKEN.match(formula_string, position)
        if token is None:
            raise ValueError(_INVALID_FORMULA_SYNTAX)
        element, count = token.groups()
        if element not in _ELEMENT_SYMBOLS:
            raise ValueError(
                "Cannot parse the molecular formula because it contains an "
                "unrecognized element symbol."
            )
        try:
            counts[element] += int(count) if count else 1
        except ValueError as error:
            raise ValueError(
                "Cannot parse the molecular formula because an atom count is too large."
            ) from error
        position = token.end()
    if not counts:
        raise ValueError(_INVALID_FORMULA_SYNTAX)
    return dict(counts)


def build_formula_string(atom_counts: dict) -> str:
    """
    Hill-system formula string: C, H, then others alphabetically; omits counts <= 0 and '1'.

    Args:
        atom_counts (dict): The atom count dictionary {"C": 10, "H": 12, "O": 6}

    Returns:
        str: The molecular formula
    """
    others = sorted(e for e in atom_counts if e not in ("C", "H"))
    element_order = ["C", "H", *others]

    parts = []
    for element in element_order:
        count = atom_counts.get(element, 0)
        if count > 0:
            parts.append(element if count == 1 else f"{element}{count}")

    return "".join(parts)


def _apply_deltas(base: dict[str, int], deltas: dict[str, int]) -> dict[str, int] | None:
    """Apply atom-count deltas; return None if any count would go negative."""
    counts = base.copy()
    for elem, d in deltas.items():
        counts[elem] = counts.get(elem, 0) + d
        if counts[elem] < 0:
            return None
    # Drop zero-count elements
    return {e: c for e, c in counts.items() if c > 0}


def _zero_out(base: dict[str, int], elements: list[str], h_delta: int) -> dict[str, int] | None:
    """Remove given elements entirely, adjusting H. Return None if nothing changed."""
    if not any(base.get(e, 0) > 0 for e in elements):
        return None  # transformation is a no-op
    counts = {e: c for e, c in base.items() if e not in elements and c > 0}
    counts["H"] = counts.get("H", 0) + h_delta
    if counts["H"] < 0:
        return None
    return counts


def gen_close_molformulas_from_seed(seed_formula: str) -> list[str]:
    """Generate chemically plausible molecular formulas near a seed.

    The returned list starts with the canonicalised seed formula itself,
    followed by neighbours reachable by small atom-count edits. Callers use
    this to restrict a candidate database, so omitting the seed would exclude
    every molecule with the target formula.
    """
    initial = get_atom_counts_from_formula(seed_formula)

    # Simple delta-based transformations
    delta_sets: list[dict[str, int]] = [
        {"C": -3, "H": -6},
        {"C": +1, "H": +2},
        {"C": -1, "H": -2},
        {"C": +2, "H": +4},
        {"C": -2, "H": -4},
        {"N": +1, "H": +1},
        {"N": -1, "H": -1},
        {"Cl": +1, "H": +1},
        {"Cl": -1, "H": -1},
        {"Br": +1, "H": +1},
        {"Br": -1, "H": -1},
        {"F": +1, "H": +1},
        {"S": +1},
        {"S": -1},
        {"P": +1},
        {"P": -1},
    ]

    candidates: list[dict[str, int] | None] = [_apply_deltas(initial, d) for d in delta_sets]

    # Structural removals
    total_halogens = sum(initial.get(x, 0) for x in ("Cl", "Br", "F"))
    candidates.append(_zero_out(initial, ["Cl", "Br", "F"], h_delta=total_halogens))
    candidates.append(_zero_out(initial, ["P"], h_delta=5))
    candidates.append(_zero_out(initial, ["S"], h_delta=4))

    # The seed itself comes first: for structure elucidation the target molecule
    # has exactly this formula, so excluding it would make the correct answer
    # unreachable from any formula-filtered candidate pool.
    seed_canonical = build_formula_string({e: c for e, c in initial.items() if c > 0})
    seen = {seed_canonical}
    results: list[str] = [seed_canonical]
    for counts in candidates:
        if counts is None:
            continue
        formula = build_formula_string(counts)
        if formula and formula not in seen:
            seen.add(formula)
            results.append(formula)
    return results


def smiles_to_molecular_formula(smiles: str) -> str:
    """Convert a SMILES string to a molecular formula."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return rdMolDescriptors.CalcMolFormula(mol)


def is_neutral_no_isotopes(smiles: str) -> bool:
    """Check if molecule is neutral and contains no isotopes"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False

        # Check for formal charges
        total_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
        if total_charge != 0:
            return False

        # Check for isotopes
        has_isotopes = any(atom.GetIsotope() != 0 for atom in mol.GetAtoms())
        return not has_isotopes
    except Exception:
        return False


def reduce_resolution_by_averaging(vector: np.ndarray, window_size: int) -> np.ndarray:
    """
    Reduces the resolution of a vector by window averaging and interpolation.

    Args:
        vector (np.ndarray): The input 1D numpy array of data.
        window_size (int): The size of the averaging window. A larger
                           window results in lower resolution.

    Returns:
        np.ndarray: A new vector with reduced resolution but the same
                    length as the input vector.
    """
    if isinstance(vector, list):
        vector = np.array(vector)

    if window_size <= 1:
        return vector

    averaged_vector = np.convolve(vector, np.ones(window_size) / window_size, mode="valid")
    original_x = np.linspace(0, 1, len(vector))
    averaged_x = np.linspace(0, 1, len(averaged_vector))
    interp_func = interp1d(averaged_x, averaged_vector, kind="linear", fill_value="extrapolate")

    # Apply the interpolation function to the original x-coordinates
    return interp_func(original_x)
