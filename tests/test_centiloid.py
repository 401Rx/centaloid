"""Unit tests for the centiloid computation module."""

import numpy as np
import pytest

from centaloid.centiloid import (
    get_tracer_params,
    list_supported_tracers,
    classify_centiloid,
    compute_centiloid,
    compute_suvr,
)
from centaloid.registration import MNI_AFFINE, MNI_SHAPE
from centaloid.atlas import (
    generate_ctx_mask,
    generate_cerebellum_mask,
)


class TestTracerDatabase:
    def test_known_tracers(self):
        for name in ("PiB", "Florbetapir", "Florbetaben", "Flutemetamol", "NAV4694"):
            assert get_tracer_params(name) is not None

    def test_case_insensitive(self):
        assert get_tracer_params("pib") is not None
        assert get_tracer_params("AMYVID") is not None

    def test_unknown_tracer(self):
        assert get_tracer_params("not-a-tracer") is None

    def test_list_supported(self):
        tracers = list_supported_tracers()
        assert len(tracers) >= 5
        assert "PiB" in tracers


class TestTracerConversion:
    def test_pib_yc_gives_zero(self):
        params = get_tracer_params("PiB")
        cl = params.suvr_to_centiloid(params.suvr_yc)
        assert abs(cl) < 1.0  # should be ~0

    def test_pib_ad_gives_100(self):
        params = get_tracer_params("PiB")
        cl = params.suvr_to_centiloid(params.suvr_ad)
        assert abs(cl - 100) < 2.0  # should be ~100

    def test_roundtrip(self):
        params = get_tracer_params("Florbetapir")
        for suvr in (0.9, 1.0, 1.2, 1.5, 2.0):
            cl = params.suvr_to_centiloid(suvr)
            suvr_back = params.centiloid_to_suvr(cl)
            assert abs(suvr_back - suvr) < 1e-6


class TestClassification:
    def test_negative(self):
        assert "Negative" in classify_centiloid(5)

    def test_indeterminate(self):
        assert "Indeterminate" in classify_centiloid(15)

    def test_moderate(self):
        assert "Moderate" in classify_centiloid(35)

    def test_positive(self):
        assert "Positive" in classify_centiloid(75)

    def test_highly_positive(self):
        assert "Highly Positive" in classify_centiloid(120)


class TestROIMasks:
    def test_ctx_mask_shape(self):
        mask = generate_ctx_mask(MNI_SHAPE, MNI_AFFINE)
        assert mask.shape == MNI_SHAPE
        assert mask.dtype == bool
        assert mask.sum() > 0

    def test_cerebellum_mask_shape(self):
        mask = generate_cerebellum_mask(MNI_SHAPE, MNI_AFFINE)
        assert mask.shape == MNI_SHAPE
        assert mask.sum() > 0

    def test_no_overlap(self):
        ctx = generate_ctx_mask(MNI_SHAPE, MNI_AFFINE)
        cb = generate_cerebellum_mask(MNI_SHAPE, MNI_AFFINE)
        overlap = (ctx & cb).sum()
        # Minimal overlap expected
        assert overlap < 100


class TestComputeSuvr:
    def test_uniform_volume(self):
        """A uniform volume should give SUVr ≈ 1.0."""
        vol = np.ones(MNI_SHAPE, dtype=np.float64) * 1000.0
        suvr, ctx_mean, ref_mean, regions = compute_suvr(vol, MNI_AFFINE)
        assert abs(suvr - 1.0) < 0.01
        assert ctx_mean > 0
        assert ref_mean > 0

    def test_compute_centiloid_uniform(self):
        vol = np.ones(MNI_SHAPE, dtype=np.float64) * 1000.0
        result = compute_centiloid(vol, MNI_AFFINE, "PiB")
        # SUVr ≈ 1.0, which is near young-control for PiB
        assert result.centiloid < 10


class TestComputeCentiloidErrors:
    def test_unknown_tracer_raises(self):
        vol = np.ones(MNI_SHAPE, dtype=np.float64)
        with pytest.raises(ValueError, match="Unknown tracer"):
            compute_centiloid(vol, MNI_AFFINE, "FakeTracer")
