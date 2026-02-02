"""Brain atlas ROI definitions for Centiloid quantification.

The standard Centiloid project defines two ROIs in MNI-152 space:
  * **CTX** – a cortical composite target region (frontal, temporal, parietal,
    posterior cingulate / precuneus).
  * **WC** (whole cerebellum) – the reference region.

Because shipping full NIfTI atlas volumes would add a large binary dependency,
this module provides *procedurally generated* approximate ROI masks in MNI space
using anatomical bounding boxes derived from the published Centiloid ROI
coordinates.  For production clinical use these should be replaced with the
official Centiloid ROI NIfTI files from the GAAIN website.

All coordinates are in **MNI-152 RAS** millimetres.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# MNI-space bounding-box ROI definitions (approximate)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BBox:
    """Axis-aligned bounding box in MNI-152 RAS mm."""
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float


# Approximate bounding boxes for cortical sub-regions composing the
# standard Centiloid CTX target VOI (union of all four).
CTX_FRONTAL = BBox(x_min=-40, x_max=40, y_min=10, y_max=70, z_min=-10, z_max=50)
CTX_TEMPORAL = BBox(x_min=-55, x_max=55, y_min=-40, y_max=10, z_min=-30, z_max=10)
CTX_PARIETAL = BBox(x_min=-45, x_max=45, y_min=-70, y_max=-10, z_min=25, z_max=65)
CTX_PCCPRECUNEUS = BBox(x_min=-15, x_max=15, y_min=-65, y_max=-30, z_min=15, z_max=50)

# Whole-cerebellum reference region
CEREBELLUM_WC = BBox(x_min=-55, x_max=55, y_min=-80, y_max=-40, z_min=-55, z_max=-25)

# All standard cortical boxes combined
CTX_BOXES: list[BBox] = [CTX_FRONTAL, CTX_TEMPORAL, CTX_PARIETAL, CTX_PCCPRECUNEUS]


# ---------------------------------------------------------------------------
# Ellipsoidal refinement for more anatomically plausible masks
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Ellipsoid:
    """Axis-aligned ellipsoid in MNI mm."""
    cx: float
    cy: float
    cz: float
    rx: float
    ry: float
    rz: float


# Approximate ellipsoidal fits
CTX_ELLIPSOID = Ellipsoid(cx=0, cy=-5, cz=20, rx=55, ry=60, rz=50)
# Cerebellum ellipsoid fitted to the MNI-152 cerebellar grey matter.
# Using the ellipsoid alone (without the bounding box) avoids sampling
# non-brain voxels in the posterior fossa.
CEREBELLUM_ELLIPSOID = Ellipsoid(cx=0, cy=-58, cz=-35, rx=48, ry=18, rz=14)


# ---------------------------------------------------------------------------
# Mask generation
# ---------------------------------------------------------------------------

def _bbox_mask(coords_mni: np.ndarray, bbox: BBox) -> np.ndarray:
    """Boolean mask for voxels inside *bbox*. ``coords_mni`` is (N, 3)."""
    return (
        (coords_mni[:, 0] >= bbox.x_min) & (coords_mni[:, 0] <= bbox.x_max) &
        (coords_mni[:, 1] >= bbox.y_min) & (coords_mni[:, 1] <= bbox.y_max) &
        (coords_mni[:, 2] >= bbox.z_min) & (coords_mni[:, 2] <= bbox.z_max)
    )


def _ellipsoid_mask(coords_mni: np.ndarray, ell: Ellipsoid) -> np.ndarray:
    """Boolean mask for voxels inside *ell*."""
    return (
        ((coords_mni[:, 0] - ell.cx) / ell.rx) ** 2 +
        ((coords_mni[:, 1] - ell.cy) / ell.ry) ** 2 +
        ((coords_mni[:, 2] - ell.cz) / ell.rz) ** 2
    ) <= 1.0


def generate_ctx_mask(
    shape: tuple[int, int, int],
    affine: np.ndarray,
    use_ellipsoid: bool = True,
) -> np.ndarray:
    """Generate the cortical target (CTX) ROI mask.

    Parameters
    ----------
    shape : (Z, Y, X)
        Volume dimensions.
    affine : (4, 4)
        Voxel → MNI-152 RAS affine.
    use_ellipsoid : bool
        If True, intersect the bounding-box union with the cortical
        ellipsoid to exclude deep white matter.

    Returns
    -------
    mask : bool ndarray of *shape*
    """
    coords = _voxel_to_mni(shape, affine)
    mask = np.zeros(coords.shape[0], dtype=bool)
    for bbox in CTX_BOXES:
        mask |= _bbox_mask(coords, bbox)
    if use_ellipsoid:
        mask &= _ellipsoid_mask(coords, CTX_ELLIPSOID)
    return mask.reshape(shape)


def generate_cerebellum_mask(
    shape: tuple[int, int, int],
    affine: np.ndarray,
    use_ellipsoid: bool = True,
) -> np.ndarray:
    """Generate the whole-cerebellum (WC) reference ROI mask.

    Uses the cerebellar ellipsoid directly rather than intersecting with
    the bounding box.  The bounding box extended well beyond the brain
    boundary in the posterior fossa, causing ~50 % of voxels to sample
    non-brain signal and deflating the reference mean.
    """
    coords = _voxel_to_mni(shape, affine)
    if use_ellipsoid:
        mask = _ellipsoid_mask(coords, CEREBELLUM_ELLIPSOID)
    else:
        mask = _bbox_mask(coords, CEREBELLUM_WC)
    return mask.reshape(shape)


def _voxel_to_mni(shape: tuple[int, int, int], affine: np.ndarray) -> np.ndarray:
    """Map every voxel index to MNI mm.  Returns (N, 3) float64."""
    kk, jj, ii = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    ijk = np.column_stack([kk.ravel(), jj.ravel(), ii.ravel(), np.ones(kk.size)])
    mni = (affine @ ijk.T).T[:, :3]
    return mni


# ---------------------------------------------------------------------------
# Region labels used in reports
# ---------------------------------------------------------------------------

REGION_LABELS: dict[str, str] = {
    "ctx_frontal": "Frontal Cortex",
    "ctx_temporal": "Lateral Temporal Cortex",
    "ctx_parietal": "Parietal Cortex",
    "ctx_pcc_precuneus": "PCC / Precuneus",
    "ctx_composite": "Cortical Composite (CTX)",
    "cerebellum_wc": "Whole Cerebellum (WC)",
}


def generate_subregion_masks(
    shape: tuple[int, int, int],
    affine: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return a dict mapping region key → boolean mask for sub-regions."""
    coords = _voxel_to_mni(shape, affine)
    masks: dict[str, np.ndarray] = {}
    ellip = _ellipsoid_mask(coords, CTX_ELLIPSOID)

    for key, bbox in [
        ("ctx_frontal", CTX_FRONTAL),
        ("ctx_temporal", CTX_TEMPORAL),
        ("ctx_parietal", CTX_PARIETAL),
        ("ctx_pcc_precuneus", CTX_PCCPRECUNEUS),
    ]:
        m = _bbox_mask(coords, bbox) & ellip
        masks[key] = m.reshape(shape)

    masks["ctx_composite"] = generate_ctx_mask(shape, affine, use_ellipsoid=True)
    masks["cerebellum_wc"] = generate_cerebellum_mask(shape, affine, use_ellipsoid=True)
    return masks
