"""Centiloid computation: SUVr calculation and Centiloid scale conversion.

The Centiloid (CL) scale was introduced by Klunk et al. (2015) to
standardise amyloid-PET quantification across tracers and centres.

    CL = 100 × (SUVr_ind − SUVr_YC) / (SUVr_AD − SUVr_AD)

where SUVr_YC and SUVr_AD are the mean SUVr values from Young-Control
and typical-AD cohorts for the given tracer.

Tracer-specific linear equations (SUVr → CL) published by the Centiloid
Working Group are implemented here.

References
----------
* Klunk WE et al. Alzheimers Dement 2015;11(1):1-15.
* Navitsky M et al. Alzheimers Dement 2018;14(12):1565-1571.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .atlas import (
    generate_ctx_mask,
    generate_cerebellum_mask,
    generate_subregion_masks,
    REGION_LABELS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tracer conversion parameters  (SUVr → CL linear equation)
#   CL = slope * SUVr + intercept
# Derived from published two-point calibration data.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TracerParams:
    """Published CL conversion parameters for a given amyloid tracer."""
    name: str
    slope: float
    intercept: float
    suvr_yc: float   # mean SUVr in young-controls
    suvr_ad: float    # mean SUVr in AD cohort
    aliases: tuple[str, ...] = ()

    def suvr_to_centiloid(self, suvr: float) -> float:
        return self.slope * suvr + self.intercept

    def centiloid_to_suvr(self, cl: float) -> float:
        return (cl - self.intercept) / self.slope


# Published conversion parameters
TRACER_DATABASE: dict[str, TracerParams] = {}

def _register(tp: TracerParams):
    key = tp.name.lower()
    TRACER_DATABASE[key] = tp
    for alias in tp.aliases:
        TRACER_DATABASE[alias.lower()] = tp

_register(TracerParams(
    name="PiB",
    slope=94.6, intercept=-94.6,  # CL = 94.6*(SUVr) − 94.6
    suvr_yc=1.009, suvr_ad=2.068,
    aliases=("pittsburgh compound b", "c-11 pib", "c11-pib", "11c-pib"),
))
_register(TracerParams(
    name="Florbetapir",
    slope=196.9, intercept=-196.03,
    suvr_yc=1.000, suvr_ad=1.508,
    aliases=("amyvid", "av-45", "av45", "18f-florbetapir", "18f-av45"),
))
_register(TracerParams(
    name="Florbetaben",
    slope=153.4, intercept=-154.87,
    suvr_yc=1.010, suvr_ad=1.662,
    aliases=("neuraceq", "bay-94-9172", "18f-florbetaben"),
))
_register(TracerParams(
    name="Flutemetamol",
    slope=121.42, intercept=-121.16,
    suvr_yc=0.998, suvr_ad=1.822,
    aliases=("vizamyl", "ge-067", "18f-flutemetamol"),
))
_register(TracerParams(
    name="NAV4694",
    slope=100.0, intercept=-100.5,
    suvr_yc=1.005, suvr_ad=2.005,
    aliases=("nav-4694", "18f-nav4694", "az-4694"),
))


def get_tracer_params(tracer_name: str) -> Optional[TracerParams]:
    """Look up tracer conversion parameters by name (case-insensitive)."""
    return TRACER_DATABASE.get(tracer_name.strip().lower())


def list_supported_tracers() -> list[str]:
    """Return a sorted list of canonical tracer names."""
    seen: set[str] = set()
    result: list[str] = []
    for tp in TRACER_DATABASE.values():
        if tp.name not in seen:
            seen.add(tp.name)
            result.append(tp.name)
    return sorted(result)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class RegionResult:
    """Quantification for a single ROI."""
    name: str
    label: str
    mean_uptake: float
    std_uptake: float
    voxel_count: int
    volume_cc: float  # mL


@dataclass
class CentiloidResult:
    """Complete Centiloid analysis result."""
    tracer: str
    suvr: float
    centiloid: float
    ctx_mean: float
    ref_mean: float
    regions: list[RegionResult] = field(default_factory=list)
    classification: str = ""
    confidence_note: str = ""

    def __post_init__(self):
        if not self.classification:
            self.classification = classify_centiloid(self.centiloid)


def classify_centiloid(cl: float) -> str:
    """Classify amyloid status based on the CL value."""
    if cl < 12:
        return "Negative (< 12 CL)"
    elif cl < 20:
        return "Indeterminate (12 – 20 CL)"
    elif cl < 50:
        return "Moderately Positive (20 – 50 CL)"
    elif cl < 100:
        return "Positive (50 – 100 CL)"
    else:
        return "Highly Positive (≥ 100 CL)"


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute_suvr(
    volume: np.ndarray,
    affine: np.ndarray,
    background_threshold: float = 0.01,
) -> tuple[float, float, float, dict[str, RegionResult]]:
    """Compute SUVr = mean(CTX) / mean(WC) and per-region stats.

    Parameters
    ----------
    volume : (Z, Y, X) ndarray  – PET data in MNI space.
    affine : (4, 4) – MNI affine.
    background_threshold : float
        Voxels with values below this fraction of the volume's non-zero mean
        are excluded from ROI statistics. This removes resampling artifacts
        (cval=0 padding) from the computation. Default 0.01 (1%).

    Returns
    -------
    suvr, ctx_mean, ref_mean, region_results
    """
    shape = volume.shape
    masks = generate_subregion_masks(shape, affine)

    voxel_vol_cc = abs(np.linalg.det(affine[:3, :3])) / 1000.0  # mm³→mL

    # Compute threshold to exclude resampling padding (cval=0) artifacts
    # Use a fraction of the mean of non-zero voxels as the cutoff
    nonzero_vals = volume[volume > 0]
    if nonzero_vals.size > 0:
        min_threshold = background_threshold * nonzero_vals.mean()
    else:
        min_threshold = 0.0

    region_results: dict[str, RegionResult] = {}
    for key, mask in masks.items():
        vals = volume[mask]
        # Exclude near-zero voxels (resampling artifacts)
        valid_vals = vals[vals > min_threshold]
        n = int(valid_vals.size)
        region_results[key] = RegionResult(
            name=key,
            label=REGION_LABELS.get(key, key),
            mean_uptake=float(valid_vals.mean()) if n > 0 else 0.0,
            std_uptake=float(valid_vals.std()) if n > 0 else 0.0,
            voxel_count=n,
            volume_cc=n * voxel_vol_cc,
        )

    ctx_mean = region_results["ctx_composite"].mean_uptake
    ref_mean = region_results["cerebellum_wc"].mean_uptake

    if ref_mean <= 0:
        logger.warning("Reference region mean is ≤ 0 – SUVr will be invalid.")
        suvr = 0.0
    else:
        suvr = ctx_mean / ref_mean

    return suvr, ctx_mean, ref_mean, region_results


def compute_centiloid(
    volume: np.ndarray,
    affine: np.ndarray,
    tracer_name: str,
) -> CentiloidResult:
    """End-to-end Centiloid computation on an MNI-space PET volume.

    Parameters
    ----------
    volume : MNI-space PET data.
    affine : MNI affine.
    tracer_name : e.g. "Florbetapir", "PiB".

    Returns
    -------
    CentiloidResult
    """
    params = get_tracer_params(tracer_name)
    if params is None:
        raise ValueError(
            f"Unknown tracer '{tracer_name}'. "
            f"Supported: {list_supported_tracers()}"
        )

    suvr, ctx_mean, ref_mean, regions = compute_suvr(volume, affine)
    cl = params.suvr_to_centiloid(suvr)

    note = ""
    if regions["cerebellum_wc"].voxel_count < 100:
        note = ("Warning: very few voxels in cerebellum reference region – "
                "registration quality may be poor.")
    if regions["ctx_composite"].voxel_count < 100:
        note += (" Warning: very few voxels in cortical target region – "
                 "registration quality may be poor.")

    return CentiloidResult(
        tracer=params.name,
        suvr=round(suvr, 4),
        centiloid=round(cl, 1),
        ctx_mean=round(ctx_mean, 4),
        ref_mean=round(ref_mean, 4),
        regions=list(regions.values()),
        confidence_note=note.strip(),
    )
