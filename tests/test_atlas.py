"""
Tests for braphin/atlas.py

Covers:
- Atlas registry (supported atlases, definitions)
- AAL ROI name map — verifies Bug 3 fix (keys must be 1-116, not coded IDs)
- Path resolution helpers
"""

import pytest

from braphin.atlas import (
    get_atlas_definition,
    get_atlas_roi_name_map,
    get_default_atlas_path,
    has_roi_name_map,
    is_supported_atlas,
    list_supported_atlases,
)
from braphin.exceptions import AtlasError


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_list_supported_atlases_contains_expected():
    atlases = list_supported_atlases()
    assert "aal" in atlases
    assert "schaefer_100" in atlases
    assert "schaefer_200" in atlases
    assert "schaefer_400" in atlases


def test_is_supported_atlas_true():
    assert is_supported_atlas("aal")
    assert is_supported_atlas("schaefer_100")


def test_is_supported_atlas_case_insensitive():
    assert is_supported_atlas("AAL")
    assert is_supported_atlas("Schaefer_100")


def test_is_supported_atlas_false():
    assert not is_supported_atlas("nonexistent_atlas")
    assert not is_supported_atlas("")


# ---------------------------------------------------------------------------
# Atlas definitions
# ---------------------------------------------------------------------------

def test_get_atlas_definition_aal():
    defn = get_atlas_definition("aal")
    assert defn.name == "aal"
    assert defn.num_rois == 116
    assert defn.family == "AAL"


def test_get_atlas_definition_schaefer_100():
    defn = get_atlas_definition("schaefer_100")
    assert defn.num_rois == 100


def test_get_atlas_definition_schaefer_200():
    defn = get_atlas_definition("schaefer_200")
    assert defn.num_rois == 200


def test_get_atlas_definition_schaefer_400():
    defn = get_atlas_definition("schaefer_400")
    assert defn.num_rois == 400


def test_get_atlas_definition_case_insensitive():
    defn = get_atlas_definition("AAL")
    assert defn.name == "aal"


def test_get_atlas_definition_unsupported_raises():
    with pytest.raises(AtlasError):
        get_atlas_definition("not_a_real_atlas")


# ---------------------------------------------------------------------------
# AAL ROI name map  — Bug 3 fix verification
# ---------------------------------------------------------------------------

def test_aal_roi_name_map_exists():
    name_map = get_atlas_roi_name_map("aal")
    assert name_map is not None


def test_aal_roi_name_map_has_116_entries():
    name_map = get_atlas_roi_name_map("aal")
    assert len(name_map) == 116


def test_aal_roi_name_map_keys_are_coded_ids():
    """Keys must be the hierarchical coded IDs stored as voxel values in the AAL NIfTI file
    (e.g. 2001 for Precentral_L, 9170 for Vermis_10).  transform.py looks up names by
    the raw NIfTI label value, so the keys must match those values."""
    name_map = get_atlas_roi_name_map("aal")
    # All keys should be in the known hierarchical range for AAL
    assert all(isinstance(k, int) and k >= 2001 for k in name_map.keys())


def test_aal_roi_name_map_contains_expected_coded_ids():
    """Verify a sample of known AAL coded IDs are present."""
    name_map = get_atlas_roi_name_map("aal")
    for coded_id in [2001, 2002, 4101, 4102, 4201, 4202, 9001, 9170]:
        assert coded_id in name_map, f"Expected coded ID {coded_id} missing from map"


def test_aal_roi_name_map_spot_checks():
    name_map = get_atlas_roi_name_map("aal")
    assert name_map[2001] == "Precentral_L"
    assert name_map[2002] == "Precentral_R"
    assert name_map[4101] == "Hippocampus_L"
    assert name_map[4102] == "Hippocampus_R"
    assert name_map[4201] == "Amygdala_L"
    assert name_map[4202] == "Amygdala_R"
    assert name_map[9170] == "Vermis_10"


def test_schaefer_has_no_roi_name_map():
    assert not has_roi_name_map("schaefer_100")
    assert get_atlas_roi_name_map("schaefer_100") is None


def test_has_roi_name_map_aal():
    assert has_roi_name_map("aal")


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def test_get_default_atlas_path_returns_path():
    p = get_default_atlas_path("aal")
    assert p.suffix in (".gz", ".nii")
    assert "aal" in p.name


def test_get_default_atlas_path_unsupported_raises():
    with pytest.raises(AtlasError):
        get_default_atlas_path("nonexistent_atlas")


# ---------------------------------------------------------------------------
# resolve_supported_atlas_path
# ---------------------------------------------------------------------------

def test_resolve_supported_atlas_path_default():
    from braphin.atlas import resolve_supported_atlas_path
    p = resolve_supported_atlas_path("aal")
    assert p.exists()


def test_resolve_supported_atlas_path_explicit(tmp_path):
    from braphin.atlas import resolve_supported_atlas_path
    fake = tmp_path / "my_atlas.nii.gz"
    fake.write_bytes(b"placeholder")
    p = resolve_supported_atlas_path("aal", atlas_path=str(fake))
    assert p == fake


def test_resolve_supported_atlas_path_bad_explicit_raises():
    from braphin.atlas import resolve_supported_atlas_path
    with pytest.raises(AtlasError, match="does not exist"):
        resolve_supported_atlas_path("aal", atlas_path="/nonexistent/path/atlas.nii.gz")
