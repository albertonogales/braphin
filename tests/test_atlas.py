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


def test_aal_roi_name_map_keys_are_1_to_116():
    """Keys must match the sequential integer labels stored in the NIfTI file."""
    name_map = get_atlas_roi_name_map("aal")
    assert set(name_map.keys()) == set(range(1, 117))


def test_aal_roi_name_map_no_old_coded_ids():
    """Old coded IDs (2001, 2101 …) must no longer be present."""
    name_map = get_atlas_roi_name_map("aal")
    for old_id in [2001, 2002, 2101, 2102, 4101, 9001]:
        assert old_id not in name_map, f"Old coded ID {old_id} still present in map"


def test_aal_roi_name_map_spot_checks():
    name_map = get_atlas_roi_name_map("aal")
    assert name_map[1] == "Precentral_L"
    assert name_map[2] == "Precentral_R"
    assert name_map[37] == "Hippocampus_L"
    assert name_map[38] == "Hippocampus_R"
    assert name_map[41] == "Amygdala_L"
    assert name_map[42] == "Amygdala_R"
    assert name_map[116] == "Vermis_10"


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
