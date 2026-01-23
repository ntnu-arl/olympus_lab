from typing import Sequence


def make_slice_if_contigious(
    indices: Sequence[int],
) -> slice | Sequence[int]:
    """Make a slice if the indices are contiguous, otherwise return the indices as is."""

    if len(indices) == 0:
        raise ValueError("Input indices cannot be empty.")
    if len(indices) == 1:
        return slice(indices[0], indices[0] + 1)

    if all([indices[i] + 1 == indices[i + 1] for i in range(len(indices) - 1)]):
        return slice(indices[0], indices[-1] + 1)

    return indices
