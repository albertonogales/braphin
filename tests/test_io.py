"""
Tests for braphin/io/tabular.py and braphin/io/nifti.py

Covers:
- NPY, CSV, TSV loading (including Bug 2 fix: BIDS TSV headers)
- load_tabular_file dispatch
- NIfTI loading, metadata extraction, and 4D validation
"""

import json

import nibabel as nib
import numpy as np
import pytest

from braphin.exceptions import BRAPHINFormatError, BRAPHINInputError
from braphin.io.nifti import get_nifti_metadata, load_nifti_file, validate_fmri_nifti
from braphin.io.tabular import (
    load_csv_file,
    load_npy_file,
    load_tabular_file,
    load_tsv_file,
)


# ---------------------------------------------------------------------------
# NPY
# ---------------------------------------------------------------------------

def test_load_npy_1d(tmp_path):
    arr = np.array([1.0, 2.0, 3.0])
    p = tmp_path / "data.npy"
    np.save(str(p), arr)
    loaded = load_npy_file(p)
    np.testing.assert_array_equal(loaded, arr)


def test_load_npy_2d(tmp_path):
    arr = np.arange(12, dtype=float).reshape(4, 3)
    p = tmp_path / "data.npy"
    np.save(str(p), arr)
    loaded = load_npy_file(p)
    assert loaded.shape == (4, 3)
    np.testing.assert_array_equal(loaded, arr)


def test_load_npy_missing_raises(tmp_path):
    with pytest.raises(BRAPHINInputError):
        load_npy_file(tmp_path / "missing.npy")


def test_load_npy_wrong_extension_raises(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("1,2,3")
    with pytest.raises(BRAPHINInputError):
        load_npy_file(p)


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def test_load_csv_numeric_no_header(tmp_path):
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    p = tmp_path / "data.csv"
    np.savetxt(str(p), arr, delimiter=",")
    loaded = load_csv_file(p)
    assert loaded.shape == (2, 3)
    np.testing.assert_allclose(loaded, arr, atol=1e-5)


def test_load_csv_with_text_header(tmp_path):
    """Header row must be skipped — no NaN row in result."""
    p = tmp_path / "data.csv"
    p.write_text("col_a,col_b,col_c\n1.0,2.0,3.0\n4.0,5.0,6.0\n")
    loaded = load_csv_file(p)
    assert loaded.shape == (2, 3), f"Expected (2,3) but got {loaded.shape}"
    assert not np.any(np.isnan(loaded))
    assert loaded[0, 0] == pytest.approx(1.0)


def test_load_csv_missing_raises(tmp_path):
    with pytest.raises(BRAPHINInputError):
        load_csv_file(tmp_path / "missing.csv")


# ---------------------------------------------------------------------------
# TSV  — Bug 2 fix verification
# ---------------------------------------------------------------------------

def test_load_tsv_numeric_no_header(tmp_path):
    arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    p = tmp_path / "data.tsv"
    np.savetxt(str(p), arr, delimiter="\t")
    loaded = load_tsv_file(p)
    assert loaded.shape == (3, 2)
    np.testing.assert_allclose(loaded, arr, atol=1e-5)


def test_load_tsv_with_bids_header(tmp_path):
    """
    Bug 2 fix: real BIDS confound TSVs have a text header row.
    The loader must skip it instead of converting it to a NaN row.
    """
    p = tmp_path / "confounds_timeseries.tsv"
    p.write_text(
        "trans_x\ttrans_y\ttrans_z\trot_x\trot_y\trot_z\n"
        "0.1\t0.2\t0.3\t0.01\t0.02\t0.03\n"
        "0.4\t0.5\t0.6\t0.04\t0.05\t0.06\n"
        "0.7\t0.8\t0.9\t0.07\t0.08\t0.09\n"
    )
    loaded = load_tsv_file(p)
    assert loaded.shape == (3, 6), f"Expected (3,6) but got {loaded.shape}"
    assert not np.any(np.isnan(loaded))
    assert loaded[0, 0] == pytest.approx(0.1)
    assert loaded[2, 5] == pytest.approx(0.09)


def test_load_tsv_bids_header_row_count(tmp_path):
    """Number of data rows must equal number of timepoints, not timepoints+1."""
    n_timepoints = 20
    header = "motion_x\tmotion_y\n"
    data_rows = "\n".join(f"{i*0.1:.3f}\t{i*0.2:.3f}" for i in range(n_timepoints))
    p = tmp_path / "confounds.tsv"
    p.write_text(header + data_rows + "\n")
    loaded = load_tsv_file(p)
    assert loaded.shape[0] == n_timepoints


def test_load_tsv_missing_raises(tmp_path):
    with pytest.raises(BRAPHINInputError):
        load_tsv_file(tmp_path / "missing.tsv")


# ---------------------------------------------------------------------------
# load_tabular_file dispatch
# ---------------------------------------------------------------------------

def test_load_tabular_dispatches_npy(tmp_path):
    arr = np.array([[1.0, 2.0]])
    p = tmp_path / "x.npy"
    np.save(str(p), arr)
    loaded = load_tabular_file(p)
    np.testing.assert_array_equal(loaded, arr)


def test_load_tabular_dispatches_csv(tmp_path):
    p = tmp_path / "x.csv"
    np.savetxt(str(p), np.eye(3), delimiter=",")
    loaded = load_tabular_file(p)
    assert loaded.shape == (3, 3)


def test_load_tabular_dispatches_tsv(tmp_path):
    p = tmp_path / "x.tsv"
    np.savetxt(str(p), np.ones((2, 2)), delimiter="\t")
    loaded = load_tabular_file(p)
    assert loaded.shape == (2, 2)


def test_load_tabular_unsupported_extension_raises(tmp_path):
    p = tmp_path / "data.xyz"
    p.write_text("not tabular")
    with pytest.raises(BRAPHINInputError):
        load_tabular_file(p)


# ---------------------------------------------------------------------------
# NIfTI
# ---------------------------------------------------------------------------

def test_load_nifti_file_returns_image(saved_fmri_path):
    img = load_nifti_file(saved_fmri_path)
    assert img is not None
    assert hasattr(img, "get_fdata")


def test_load_nifti_file_missing_raises(tmp_path):
    with pytest.raises(BRAPHINInputError):
        load_nifti_file(tmp_path / "missing.nii.gz")


def test_load_nifti_file_wrong_extension_raises(tmp_path):
    p = tmp_path / "data.txt"
    p.write_text("not a nifti")
    with pytest.raises(BRAPHINInputError):
        load_nifti_file(p)


def test_get_nifti_metadata_keys(fmri_image):
    meta = get_nifti_metadata(fmri_image)
    for key in ("shape", "ndim", "affine", "zooms", "axis_codes"):
        assert key in meta


def test_get_nifti_metadata_values(fmri_image):
    meta = get_nifti_metadata(fmri_image)
    assert meta["ndim"] == 4
    assert len(meta["shape"]) == 4
    assert meta["affine"].shape == (4, 4)


def test_validate_fmri_nifti_passes_4d(fmri_image):
    validate_fmri_nifti(fmri_image)   # must not raise


def test_validate_fmri_nifti_rejects_3d():
    img_3d = nib.Nifti1Image(np.zeros((4, 4, 4)), np.eye(4))
    with pytest.raises(BRAPHINFormatError):
        validate_fmri_nifti(img_3d)


# ---------------------------------------------------------------------------
# Extension mismatch
# ---------------------------------------------------------------------------

def test_load_csv_wrong_extension_raises(tmp_path):
    p = tmp_path / "data.tsv"
    p.write_text("1\t2\n3\t4\n")
    with pytest.raises(BRAPHINInputError, match="csv"):
        load_csv_file(p)


def test_load_tsv_wrong_extension_raises(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("1,2\n3,4\n")
    with pytest.raises(BRAPHINInputError, match="tsv"):
        load_tsv_file(p)


def test_load_npy_corrupt_raises(tmp_path):
    p = tmp_path / "corrupt.npy"
    p.write_bytes(b"this is not a npy file at all!!!")
    with pytest.raises(BRAPHINFormatError):
        load_npy_file(p)


def test_load_nifti_corrupt_raises(tmp_path):
    p = tmp_path / "bad.nii"
    p.write_bytes(b"not a nifti file")
    with pytest.raises(BRAPHINFormatError):
        load_nifti_file(p)
