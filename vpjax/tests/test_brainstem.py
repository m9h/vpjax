"""Tests for brainstem nuclei identification from MELODIC ICA.

Uses atlas ROIs (Brainstem Navigator or synthetic) to identify
ICA components that map to brainstem nuclei, then extracts their
time courses for vpjax physiological modeling.
"""

import jax.numpy as jnp
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Brainstem atlas ROIs
# ---------------------------------------------------------------------------

class TestBrainstemAtlas:
    """Brainstem nuclei ROI definitions."""

    def test_nuclei_defined(self):
        """Should define standard brainstem nuclei."""
        from vpjax.brainstem.atlas import NUCLEI
        assert "LC" in NUCLEI       # locus coeruleus
        assert "DR" in NUCLEI       # dorsal raphe
        assert "MR" in NUCLEI       # median raphe
        assert "NTS" in NUCLEI      # nucleus tractus solitarius
        assert "PAG" in NUCLEI      # periaqueductal gray
        assert "VTA" in NUCLEI      # ventral tegmental area

    def test_create_synthetic_atlas(self):
        """Should create synthetic brainstem ROIs for testing."""
        from vpjax.brainstem.atlas import create_synthetic_atlas
        atlas = create_synthetic_atlas(shape=(91, 109, 91))
        assert isinstance(atlas, dict)
        assert "LC" in atlas
        assert atlas["LC"].shape == (91, 109, 91)
        # LC ROI should have nonzero voxels
        assert np.sum(atlas["LC"] > 0) > 0

    def test_roi_coordinates(self):
        """Each nucleus should have MNI coordinates."""
        from vpjax.brainstem.atlas import NUCLEI
        for name, info in NUCLEI.items():
            assert "mni_x" in info
            assert "mni_y" in info
            assert "mni_z" in info
            assert "radius_mm" in info


# ---------------------------------------------------------------------------
# ICA component → brainstem nuclei matching
# ---------------------------------------------------------------------------

class TestICAIdentification:
    """Match MELODIC ICA components to brainstem nuclei."""

    def test_spatial_overlap(self):
        """Should compute overlap between an IC map and an ROI."""
        from vpjax.brainstem.ica_identify import spatial_overlap
        # Synthetic IC map with a peak in the brainstem
        ic_map = np.zeros((91, 109, 91))
        ic_map[45:50, 50:55, 20:25] = 3.0  # z-scored blob
        roi = np.zeros((91, 109, 91))
        roi[46:49, 51:54, 21:24] = 1.0  # overlapping ROI
        overlap = spatial_overlap(ic_map, roi)
        assert overlap > 0.0
        assert overlap <= 1.0

    def test_no_overlap(self):
        """Non-overlapping IC and ROI should give ~0 overlap."""
        from vpjax.brainstem.ica_identify import spatial_overlap
        ic_map = np.zeros((91, 109, 91))
        ic_map[10:15, 10:15, 10:15] = 5.0
        roi = np.zeros((91, 109, 91))
        roi[80:85, 80:85, 80:85] = 1.0
        assert spatial_overlap(ic_map, roi) < 0.01

    def test_identify_brainstem_components(self):
        """Should identify which ICA components map to brainstem nuclei."""
        from vpjax.brainstem.ica_identify import identify_brainstem_components
        from vpjax.brainstem.atlas import create_synthetic_atlas

        shape = (91, 109, 91)
        n_components = 20
        atlas = create_synthetic_atlas(shape)

        # Create fake IC maps: one overlapping LC, one overlapping DR
        ic_maps = np.random.randn(n_components, *shape) * 0.1
        # Put a strong signal in the LC region for component 5
        lc_mask = atlas["LC"] > 0
        ic_maps[5][lc_mask] = 5.0
        # Put a strong signal in the DR region for component 12
        dr_mask = atlas["DR"] > 0
        ic_maps[12][dr_mask] = 4.0

        matches = identify_brainstem_components(ic_maps, atlas)
        assert isinstance(matches, dict)
        # Should find LC in component 5
        assert "LC" in matches
        assert matches["LC"]["component_idx"] == 5
        # Should find DR in component 12
        assert "DR" in matches
        assert matches["DR"]["component_idx"] == 12

    def test_overlap_threshold(self):
        """Weak overlaps should not produce matches."""
        from vpjax.brainstem.ica_identify import identify_brainstem_components
        from vpjax.brainstem.atlas import create_synthetic_atlas

        shape = (91, 109, 91)
        atlas = create_synthetic_atlas(shape)
        # All-noise IC maps — no real brainstem signal
        ic_maps = np.random.randn(10, *shape) * 0.5
        matches = identify_brainstem_components(ic_maps, atlas, threshold=2.0)
        # Should find few or no matches with high threshold
        assert len(matches) <= 3  # noise might occasionally hit


# ---------------------------------------------------------------------------
# Time course extraction
# ---------------------------------------------------------------------------

class TestTimeCourseExtraction:
    """Extract brainstem nuclei time courses from ICA."""

    def test_extract_timecourse(self):
        """Should extract mixing matrix row for identified component."""
        from vpjax.brainstem.extract import extract_timecourses
        n_components = 20
        n_timepoints = 300
        mixing = np.random.randn(n_timepoints, n_components)

        matches = {
            "LC": {"component_idx": 5, "overlap": 0.8},
            "DR": {"component_idx": 12, "overlap": 0.6},
        }

        timecourses = extract_timecourses(mixing, matches)
        assert "LC" in timecourses
        assert "DR" in timecourses
        assert timecourses["LC"].shape == (n_timepoints,)
        assert timecourses["DR"].shape == (n_timepoints,)
        # Should be the actual mixing matrix columns
        np.testing.assert_array_equal(timecourses["LC"], mixing[:, 5])

    def test_lc_to_vpjax_ne(self):
        """LC time course should connect to vpjax NE/vasomotion model."""
        from vpjax.brainstem.extract import lc_timecourse_to_ne
        # Simulate an LC time course with slow oscillation
        t = np.linspace(0, 300, 300)
        lc_tc = np.sin(2 * np.pi * 0.02 * t)  # ~0.02 Hz
        ne = lc_timecourse_to_ne(lc_tc, tr=1.0)
        assert ne.shape == lc_tc.shape
        # NE should be smoothed version of LC, anti-correlated
        # (LC activity → NE release → vasoconstriction)
        assert float(jnp.std(jnp.array(ne))) > 0.01
