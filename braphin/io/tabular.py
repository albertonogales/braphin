"""
Tabular file I/O helpers for BRAPHIN.

Loads auxiliary and derived tabular files used by the pipeline:
- ``.tsv`` — BIDS-format confound matrices, event files, etc.
- ``.csv`` — comma-separated numeric arrays.
- ``.npy`` — NumPy binary arrays.

This module is responsible only for loading files correctly; interpretation
of their content (e.g. confound regression, event modelling) is handled by
later pipeline stages.
"""

from pathlib import Path

import numpy as np

from ..exceptions import BRAPHINFormatError, BRAPHINInputError


def load_npy_file(file_path: str | Path) -> np.ndarray:
    """Load a ``.npy`` file and return its contents as a NumPy array."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise BRAPHINInputError(f"File not found: {file_path}")

    if file_path.suffix.lower() != ".npy":
        raise BRAPHINInputError(f"Expected a .npy file, got: {file_path.name}")

    try:
        data = np.load(file_path, allow_pickle=True)
    except Exception as exc:
        raise BRAPHINFormatError(f"Failed to load .npy file: {file_path}") from exc

    return np.asarray(data)


def load_delimited_file(file_path: str | Path, delimiter: str) -> np.ndarray:
    """
    Load a delimited text file (``.csv`` or ``.tsv``) as a numeric array.

    Automatically detects and skips a text header row (e.g. BIDS confound
    files that start with ``trans_x\\ttrans_y\\t…``). Columns with NaN values
    (e.g. derivative columns at the start of a run) are preserved as-is for
    downstream stages to handle.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise BRAPHINInputError(f"File not found: {file_path}")

    try:
        # Peek at the first line to detect a text header.
        with open(file_path, encoding="utf-8") as f:
            first_line = f.readline()

        # If the first cell cannot be parsed as a float, the row is a header.
        first_cell = first_line.split(delimiter)[0].strip()
        has_header = True
        try:
            float(first_cell)
            has_header = False
        except ValueError:
            has_header = True

        skip_header = 1 if has_header else 0
        data = np.genfromtxt(
            file_path,
            delimiter=delimiter,
            dtype=float,
            autostrip=True,
            skip_header=skip_header,
        )
    except (BRAPHINInputError, BRAPHINFormatError):
        raise
    except Exception as exc:
        raise BRAPHINFormatError(f"Failed to load tabular file: {file_path}") from exc

    return data


def load_csv_file(file_path: str | Path) -> np.ndarray:
    """Load a ``.csv`` file as a numeric array."""
    file_path = Path(file_path)

    if file_path.suffix.lower() != ".csv":
        raise BRAPHINInputError(f"Expected a .csv file, got: {file_path.name}")

    return load_delimited_file(file_path, delimiter=",")


def load_tsv_file(file_path: str | Path) -> np.ndarray:
    """Load a ``.tsv`` file as a numeric array."""
    file_path = Path(file_path)

    if file_path.suffix.lower() != ".tsv":
        raise BRAPHINInputError(f"Expected a .tsv file, got: {file_path.name}")

    return load_delimited_file(file_path, delimiter="\t")


def load_tabular_file(file_path: str | Path) -> np.ndarray:
    """
    Dispatch to the appropriate loader based on file extension.

    Supported extensions: ``.csv``, ``.tsv``, ``.npy``.
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        return load_csv_file(file_path)

    if suffix == ".tsv":
        return load_tsv_file(file_path)

    if suffix == ".npy":
        return load_npy_file(file_path)

    raise BRAPHINInputError(
        f"Unsupported tabular file extension: {file_path.name}. "
        "Supported extensions: .csv, .tsv, .npy"
    )
