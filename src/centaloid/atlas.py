"""Brain atlas ROI definitions for Centiloid quantification.

The standard Centiloid project (Klunk et al. 2015) defines two ROIs in MNI-152 space:
  * **CTX** – a cortical composite target region derived from AD-minus-YC PiB
    difference images, thresholded at 1.05 SUVr.
  * **WC** (whole cerebellum) – the reference region.

This module supports loading the **official GAAIN Centiloid VOI NIfTI files**
when available. If the official masks are not found, procedurally generated
approximate masks are used as a fallback (with a warning).

For accurate clinical results, download the official VOI files from:
    https://www.gaain.org/centiloid-project
    → Download "Centiloid_Std_VOI.zip"
    → Extract and place the NIfTI files in the 'data' directory.

References
----------
* Klunk WE et al. Alzheimers Dement 2015;11(1):1-15.
* GAAIN Centiloid Project: https://www.gaain.org/centiloid-project
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data directory for official VOI files
# ---------------------------------------------------------------------------

# Default location: src/centaloid/data/
_MODULE_DIR = Path(__file__).parent
DATA_DIR = _MODULE_DIR / "data"

# Environment variable override for VOI directory
VOI_DIR_ENV = "CENTILOID_VOI_DIR"

# Expected filenames for GAAIN VOI files (common naming conventions)
CTX_VOI_NAMES = [
    "voi_ctx_2mm.nii",
    "voi_ctx_2mm.nii.gz",
    "CTX_VOI.nii",
    "CTX_VOI.nii.gz",
    "ctx.nii",
    "ctx.nii.gz",
    "Centiloid_Ctx_VOI.nii",
    "Centiloid_Ctx_VOI.nii.gz",
]

WC_VOI_NAMES = [
    "voi_WhlCbl_2mm.nii",      # Official GAAIN filename
    "voi_WhlCbl_2mm.nii.gz",
    "voi_wc_2mm.nii",
    "voi_wc_2mm.nii.gz",
    "WC_VOI.nii",
    "WC_VOI.nii.gz",
    "wc.nii",
    "wc.nii.gz",
    "whole_cerebellum.nii",
    "whole_cerebellum.nii.gz",
    "Centiloid_WC_VOI.nii",
    "Centiloid_WC_VOI.nii.gz",
    "CerebellumWholeMask.nii",
    "CerebellumWholeMask.nii.gz",
]


def get_voi_directory() -> Path:
    """Return the directory containing VOI files."""
    env_dir = os.environ.get(VOI_DIR_ENV)
    if env_dir:
        return Path(env_dir)
    return DATA_DIR


def _find_voi_file(candidates: list[str]) -> Optional[Path]:
    """Search for a VOI file from a list of candidate names."""
    voi_dir = get_voi_directory()
    if not voi_dir.exists():
        return None
    for name in candidates:
        path = voi_dir / name
        if path.exists():
            return path
    return None


# ---------------------------------------------------------------------------
# VOI loading state
# ---------------------------------------------------------------------------

_LOADED_CTX_MASK: Optional[np.ndarray] = None
_LOADED_WC_MASK: Optional[np.ndarray] = None
_LOADED_VOI_AFFINE: Optional[np.ndarray] = None
_USING_OFFICIAL_VOI: bool = False


def load_official_voi_files() -> bool:
    """Attempt to load official GAAIN VOI NIfTI files.

    Returns True if official VOI files were loaded successfully.
    """
    global _LOADED_CTX_MASK, _LOADED_WC_MASK, _LOADED_VOI_AFFINE, _USING_OFFICIAL_VOI

    try:
        import nibabel as nib
    except ImportError:
        logger.warning("nibabel not installed - cannot load NIfTI VOI files")
        return False

    ctx_path = _find_voi_file(CTX_VOI_NAMES)
    wc_path = _find_voi_file(WC_VOI_NAMES)

    if ctx_path is None or wc_path is None:
        missing = []
        if ctx_path is None:
            missing.append("CTX")
        if wc_path is None:
            missing.append("WC")
        logger.warning(
            "Official GAAIN VOI files not found (%s). "
            "Using procedural fallback masks. "
            "For accurate results, download from: https://www.gaain.org/centiloid-project",
            ", ".join(missing)
        )
        return False

    try:
        ctx_img = nib.load(ctx_path)
        wc_img = nib.load(wc_path)

        _LOADED_CTX_MASK = np.asarray(ctx_img.dataobj) > 0
        _LOADED_WC_MASK = np.asarray(wc_img.dataobj) > 0
        _LOADED_VOI_AFFINE = ctx_img.affine
        _USING_OFFICIAL_VOI = True

        ctx_voxels = int(_LOADED_CTX_MASK.sum())
        wc_voxels = int(_LOADED_WC_MASK.sum())

        logger.info(
            "Loaded official GAAIN VOI files: CTX=%s (%d voxels), WC=%s (%d voxels)",
            ctx_path.name, ctx_voxels, wc_path.name, wc_voxels
        )
        return True

    except Exception as e:
        logger.error("Failed to load VOI files: %s", e)
        return False


def is_using_official_voi() -> bool:
    """Check if official GAAIN VOI files are loaded."""
    return _USING_OFFICIAL_VOI


def get_voi_status() -> str:
    """Return a human-readable status of VOI loading."""
    if _USING_OFFICIAL_VOI:
        return "Official GAAIN VOI files loaded"
    return "Using procedural fallback masks (approximate)"


# ---------------------------------------------------------------------------
# MNI-space bounding-box ROI definitions (fallback)
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
CEREBELLUM_ELLIPSOID = Ellipsoid(cx=0, cy=-60, cz=-38, rx=52, ry=22, rz=18)


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


def _voxel_to_mni(shape: tuple[int, int, int], affine: np.ndarray) -> np.ndarray:
    """Map every voxel index to MNI mm.  Returns (N, 3) float64."""
    kk, jj, ii = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    ijk = np.column_stack([kk.ravel(), jj.ravel(), ii.ravel(), np.ones(kk.size)])
    mni = (affine @ ijk.T).T[:, :3]
    return mni


def _resample_mask_to_target(
    source_mask: np.ndarray,
    source_affine: np.ndarray,
    target_shape: Tuple[int, int, int],
    target_affine: np.ndarray,
) -> np.ndarray:
    """Resample a binary mask from source space to target space.

    Uses nearest-neighbor interpolation to preserve binary values.
    """
    from scipy.ndimage import affine_transform

    # Compute the transform from target voxels to source voxels
    # target_vox -> world -> source_vox
    target_to_world = target_affine
    world_to_source = np.linalg.inv(source_affine)
    target_to_source = world_to_source @ target_to_world

    # affine_transform expects the inverse (source coords for each target coord)
    resampled = affine_transform(
        source_mask.astype(np.float32),
        target_to_source[:3, :3],
        offset=target_to_source[:3, 3],
        output_shape=target_shape,
        order=0,  # nearest neighbor
        mode='constant',
        cval=0.0,
    )
    return resampled > 0.5


def generate_ctx_mask(
    shape: tuple[int, int, int],
    affine: np.ndarray,
    use_ellipsoid: bool = True,
) -> np.ndarray:
    """Generate the cortical target (CTX) ROI mask.

    If official GAAIN VOI files are loaded, resamples the official mask.
    Otherwise, uses procedural bounding box + ellipsoid approximation.

    Parameters
    ----------
    shape : (Z, Y, X)
        Volume dimensions.
    affine : (4, 4)
        Voxel -> MNI-152 RAS affine.
    use_ellipsoid : bool
        If True (and using fallback), intersect the bounding-box union with
        the cortical ellipsoid to exclude deep white matter.

    Returns
    -------
    mask : bool ndarray of *shape*
    """
    # Use official VOI if loaded
    if _USING_OFFICIAL_VOI and _LOADED_CTX_MASK is not None and _LOADED_VOI_AFFINE is not None:
        return _resample_mask_to_target(
            _LOADED_CTX_MASK, _LOADED_VOI_AFFINE, shape, affine
        )

    # Fallback: procedural mask
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

    If official GAAIN VOI files are loaded, resamples the official mask.
    Otherwise, uses procedural bounding box + ellipsoid approximation.
    """
    # Use official VOI if loaded
    if _USING_OFFICIAL_VOI and _LOADED_WC_MASK is not None and _LOADED_VOI_AFFINE is not None:
        return _resample_mask_to_target(
            _LOADED_WC_MASK, _LOADED_VOI_AFFINE, shape, affine
        )

    # Fallback: procedural mask
    coords = _voxel_to_mni(shape, affine)
    mask = _bbox_mask(coords, CEREBELLUM_WC)
    if use_ellipsoid:
        mask &= _ellipsoid_mask(coords, CEREBELLUM_ELLIPSOID)
    return mask.reshape(shape)


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

# Reference regions used for SUVr normalization.
REFERENCE_REGION_KEYS: tuple[str, ...] = ("cerebellum_wc",)


def generate_subregion_masks(
    shape: tuple[int, int, int],
    affine: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return a dict mapping region key -> boolean mask for sub-regions.

    Note: Sub-region breakdown is only available with procedural masks.
    When using official GAAIN VOI, only ctx_composite and cerebellum_wc
    are accurate; sub-regions use procedural approximations.
    """
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

    # Use official VOI for composite and cerebellum if available
    masks["ctx_composite"] = generate_ctx_mask(shape, affine, use_ellipsoid=True)
    masks["cerebellum_wc"] = generate_cerebellum_mask(shape, affine, use_ellipsoid=True)
    return masks


# ---------------------------------------------------------------------------
# Initialization: attempt to load official VOI files at import time
# ---------------------------------------------------------------------------

# Try to load official VOI files when module is imported
load_official_voi_files()
