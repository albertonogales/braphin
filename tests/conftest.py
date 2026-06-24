"""
Shared fixtures for the braphin test suite.

All fixtures use synthetic data only — no external files required.
Spatial shape is kept small (8x8x8) and T=50 so the full suite runs quickly.
"""

import dataclasses

import nibabel as nib
import numpy as np
import pytest

from braphin.config import DenoiseConfig, PreprocessConfig
from braphin.denoise import DenoiseBRAPHINData
from braphin.importBRAPHINData import BRAPHINInputBundle
from braphin.io.nifti import get_nifti_metadata
from braphin.preprocess import PreprocessBRAPHINData
from braphin.transform import build_synthetic_atlas

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SPATIAL = (8, 8, 8)
T = 50
N_ROIS = 4


# ---------------------------------------------------------------------------
# Raw data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def spatial_shape():
    return SPATIAL


@pytest.fixture(scope="session")
def num_timepoints():
    return T


@pytest.fixture(scope="session")
def fmri_array():
    rng = np.random.default_rng(42)
    # Realistic signal range: mean ~700, std ~100 (arbitrary units)
    return (rng.random((*SPATIAL, T)) * 200.0 + 600.0).astype(np.float32)


@pytest.fixture(scope="session")
def fmri_affine():
    # 2 mm isotropic affine (4x4 homogeneous)
    return np.diag([2.0, 2.0, 2.0, 1.0]).astype(np.float64)


@pytest.fixture(scope="session")
def fmri_image(fmri_array, fmri_affine):
    img = nib.Nifti1Image(fmri_array, fmri_affine)
    img.header.set_zooms((2.0, 2.0, 2.0, 2.0))   # TR = 2 s
    return img


@pytest.fixture(scope="session")
def saved_fmri_path(fmri_image, tmp_path_factory):
    p = tmp_path_factory.mktemp("fmri") / "synthetic.nii.gz"
    nib.save(fmri_image, str(p))
    return p


# ---------------------------------------------------------------------------
# Pipeline bundles  (each stage wraps the previous one)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def input_bundle(fmri_image, saved_fmri_path):
    meta = get_nifti_metadata(fmri_image)
    return BRAPHINInputBundle(
        fmri_path=str(saved_fmri_path),
        fmri_image=fmri_image,
        fmri_metadata=meta,
        auxiliary_files={},
    )


@pytest.fixture(scope="session")
def preprocess_bundle(input_bundle):
    cfg = PreprocessConfig(
        apply_motion_correction=False,
        apply_slice_timing=False,
        apply_outlier_detection=False,
        apply_voxel_zscore=False,
        apply_smoothing=False,
    )
    return PreprocessBRAPHINData(input_bundle, cfg).run()


@pytest.fixture(scope="session")
def denoise_bundle(preprocess_bundle):
    cfg = DenoiseConfig(
        regress_confounds=False,
        apply_scrubbing=False,
        apply_bandpass=False,
    )
    return DenoiseBRAPHINData(preprocess_bundle, cfg).run()


# ---------------------------------------------------------------------------
# Atlas
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def atlas_array(spatial_shape):
    return build_synthetic_atlas(spatial_shape, num_rois=N_ROIS)
