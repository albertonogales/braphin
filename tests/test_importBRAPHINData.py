"""
Tests for braphin/importBRAPHINData.py

Covers:
- Successful load returns a complete BRAPHINInputBundle
- Metadata fields are present and correct
- File-not-found and wrong-extension errors
- Auxiliary file loading (CSV, JSON)
"""

import json

import numpy as np
import pytest

from braphin.exceptions import BRAPHINInputError
from braphin.importBRAPHINData import InputBRAPHINData, BRAPHINInputBundle


# ---------------------------------------------------------------------------
# Basic load
# ---------------------------------------------------------------------------

def test_load_returns_bundle(saved_fmri_path):
    bundle = InputBRAPHINData(saved_fmri_path).load()
    assert isinstance(bundle, BRAPHINInputBundle)


def test_load_bundle_fmri_image_present(saved_fmri_path):
    bundle = InputBRAPHINData(saved_fmri_path).load()
    assert bundle.fmri_image is not None


def test_load_bundle_fmri_path_stored(saved_fmri_path):
    bundle = InputBRAPHINData(saved_fmri_path).load()
    assert bundle.fmri_path == str(saved_fmri_path)


def test_load_bundle_metadata_present(saved_fmri_path):
    bundle = InputBRAPHINData(saved_fmri_path).load()
    assert bundle.fmri_metadata is not None
    assert bundle.fmri_metadata["ndim"] == 4


def test_load_bundle_shape_matches(saved_fmri_path, spatial_shape, num_timepoints):
    bundle = InputBRAPHINData(saved_fmri_path).load()
    shape = bundle.fmri_metadata["shape"]
    assert tuple(shape[:3]) == spatial_shape
    assert shape[3] == num_timepoints


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_load_missing_file_raises(tmp_path):
    with pytest.raises(BRAPHINInputError):
        InputBRAPHINData(tmp_path / "missing.nii.gz").load()


def test_load_wrong_extension_raises(tmp_path):
    p = tmp_path / "scan.txt"
    p.write_text("not a nifti")
    with pytest.raises(BRAPHINInputError):
        InputBRAPHINData(p).load()


def test_load_missing_auxiliary_raises(saved_fmri_path, tmp_path):
    with pytest.raises(BRAPHINInputError):
        InputBRAPHINData(
            saved_fmri_path,
            auxiliary_paths=[tmp_path / "missing.csv"],
        ).load()


# ---------------------------------------------------------------------------
# Auxiliary files
# ---------------------------------------------------------------------------

def test_load_with_csv_auxiliary(saved_fmri_path, tmp_path, num_timepoints):
    arr = np.random.rand(num_timepoints, 6).astype(np.float64)
    p = tmp_path / "confounds.csv"
    np.savetxt(str(p), arr, delimiter=",")
    bundle = InputBRAPHINData(saved_fmri_path, auxiliary_paths=[p]).load()
    assert "confounds.csv" in bundle.auxiliary_files
    assert isinstance(bundle.auxiliary_files["confounds.csv"], np.ndarray)


def test_load_with_json_auxiliary(saved_fmri_path, tmp_path):
    p = tmp_path / "metadata.json"
    p.write_text(json.dumps({"tr": 2.0, "subject": "sub-01"}))
    bundle = InputBRAPHINData(saved_fmri_path, auxiliary_paths=[p]).load()
    assert "metadata.json" in bundle.auxiliary_files
    # JSON is kept as raw text
    assert '"tr"' in bundle.auxiliary_files["metadata.json"]


def test_load_with_npy_auxiliary(saved_fmri_path, tmp_path, num_timepoints):
    arr = np.random.rand(num_timepoints, 3).astype(np.float32)
    p = tmp_path / "confounds.npy"
    np.save(str(p), arr)
    bundle = InputBRAPHINData(saved_fmri_path, auxiliary_paths=[p]).load()
    assert "confounds.npy" in bundle.auxiliary_files


def test_load_no_auxiliary_empty_dict(saved_fmri_path):
    bundle = InputBRAPHINData(saved_fmri_path).load()
    assert bundle.auxiliary_files == {}
