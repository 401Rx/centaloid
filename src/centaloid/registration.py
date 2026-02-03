"""Affine spatial normalisation of PET volumes to MNI-152 space.

This module provides a lightweight rigid + affine registration using
intensity-based optimisation (normalised cross-correlation) with scipy.
It is intentionally simple so that the package has no hard dependency on
large neuroimaging toolkits (SPM, FSL, ANTs).

For clinical-grade results replace :func:`register_to_mni` with a call to
an external tool (e.g. ``antsRegistrationSyNQuick``).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.ndimage import affine_transform, gaussian_filter
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MNI-152 template stub (2 mm resolution, 91×109×91)
# ---------------------------------------------------------------------------

MNI_SHAPE = (91, 109, 91)
MNI_VOXEL_SIZE = (2.0, 2.0, 2.0)

# Standard MNI affine (2 mm, RAS, origin at AC)
MNI_AFFINE = np.array([
    [2.0,  0.0,  0.0, -90.0],
    [0.0,  2.0,  0.0, -126.0],
    [0.0,  0.0,  2.0, -72.0],
    [0.0,  0.0,  0.0,   1.0],
])


def _synthetic_mni_template() -> np.ndarray:
    """Create an approximate MNI brain-shaped template for registration.

    This produces a smooth ellipsoidal "brain" in MNI space that is
    sufficient for gross affine alignment of PET data.  Real applications
    should load the actual MNI-152 PET template.
    """
    kk, jj, ii = np.mgrid[0:MNI_SHAPE[0], 0:MNI_SHAPE[1], 0:MNI_SHAPE[2]]
    ijk1 = np.column_stack([kk.ravel(), jj.ravel(), ii.ravel(),
                            np.ones(kk.size)])
    mni = (MNI_AFFINE @ ijk1.T).T[:, :3]

    # Ellipsoidal brain model (center ~ [0, -10, 5])
    val = (
        (mni[:, 0] / 65) ** 2 +
        ((mni[:, 1] + 10) / 80) ** 2 +
        ((mni[:, 2] - 5) / 60) ** 2
    )
    brain = np.zeros(kk.size)
    brain[val <= 1.0] = 1.0
    brain = brain.reshape(MNI_SHAPE)
    brain = gaussian_filter(brain, sigma=3)
    return brain


# ---------------------------------------------------------------------------
# Registration cost function
# ---------------------------------------------------------------------------

def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Negative normalised cross-correlation (to minimise)."""
    a_flat = a.ravel().astype(np.float64)
    b_flat = b.ravel().astype(np.float64)
    a_m = a_flat - a_flat.mean()
    b_m = b_flat - b_flat.mean()
    num = np.dot(a_m, b_m)
    den = np.sqrt(np.dot(a_m, a_m) * np.dot(b_m, b_m))
    if den < 1e-12:
        return 0.0
    return -num / den


def _params_to_affine(params: np.ndarray) -> np.ndarray:
    """Convert 12 parameters to a 4×4 affine matrix.

    params = [tx, ty, tz, rx, ry, rz, sx, sy, sz, shxy, shxz, shyz]
    """
    tx, ty, tz = params[0:3]
    rx, ry, rz = params[3:6]
    sx, sy, sz = params[6:9]
    shxy, shxz, shyz = params[9:12]

    # Rotation matrices
    cx, sx_ = np.cos(rx), np.sin(rx)
    cy, sy_ = np.cos(ry), np.sin(ry)
    cz, sz_ = np.cos(rz), np.sin(rz)

    Rx = np.array([[1, 0, 0], [0, cx, -sx_], [0, sx_, cx]])
    Ry = np.array([[cy, 0, sy_], [0, 1, 0], [-sy_, 0, cy]])
    Rz = np.array([[cz, -sz_, 0], [sz_, cz, 0], [0, 0, 1]])

    R = Rz @ Ry @ Rx

    S = np.diag([sx, sy, sz])
    Sh = np.array([[1, shxy, shxz], [0, 1, shyz], [0, 0, 1]])

    M = np.eye(4)
    M[0:3, 0:3] = R @ S @ Sh
    M[0:3, 3] = [tx, ty, tz]
    return M


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate_affine_to_mni(
    source_data: np.ndarray,
    source_affine: np.ndarray,
    max_iter: int = 200,
) -> np.ndarray:
    """Estimate a 4×4 affine that maps source voxels to MNI voxels.

    The optimisation minimises NCC between the source (resampled into MNI
    space) and the synthetic MNI template.

    Returns
    -------
    transform : (4, 4) ndarray
        Maps **source voxel** indices to **MNI voxel** indices.
    """
    template = _synthetic_mni_template()
    source_smooth = gaussian_filter(source_data.astype(np.float64), sigma=2)

    # Normalise intensities
    s_max = source_smooth.max() or 1.0
    t_max = template.max() or 1.0
    src_norm = source_smooth / s_max
    tpl_norm = template / t_max

    inv_mni = np.linalg.inv(MNI_AFFINE)

    def _cost(params):
        M_world = _params_to_affine(params)
        # source_vox → world → MNI_world → MNI_vox
        combined = inv_mni @ M_world @ source_affine
        inv_combined = np.linalg.inv(combined)
        resampled = affine_transform(
            src_norm, inv_combined[0:3, 0:3], offset=inv_combined[0:3, 3],
            output_shape=MNI_SHAPE, order=1, mode="constant", cval=0.0,
        )
        return _ncc(resampled, tpl_norm)

    # Initial parameters: identity (no translation/rotation, scale=1)
    x0 = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=np.float64)

    logger.info("Starting affine registration to MNI space (%d max iter)…", max_iter)
    result = minimize(_cost, x0, method="Powell",
                      options={"maxiter": max_iter, "ftol": 1e-6})
    logger.info("Registration finished: NCC=%.4f, nfev=%d",
                -result.fun, result.nfev)

    M_world = _params_to_affine(result.x)
    transform = inv_mni @ M_world @ source_affine
    return transform


def resample_to_mni(
    source_data: np.ndarray,
    source_affine: np.ndarray,
    transform: Optional[np.ndarray] = None,
    order: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample *source_data* into MNI-152 space (2 mm).

    Parameters
    ----------
    source_data : (Z, Y, X) ndarray
    source_affine : (4, 4) voxel→world
    transform : (4, 4) source-voxel→MNI-voxel (if None, estimated).
    order : int  Spline interpolation order.

    Returns
    -------
    (mni_data, MNI_AFFINE)
    """
    if transform is None:
        transform = estimate_affine_to_mni(source_data, source_affine)

    inv_transform = np.linalg.inv(transform)
    mni_data = affine_transform(
        source_data.astype(np.float64),
        inv_transform[0:3, 0:3],
        offset=inv_transform[0:3, 3],
        output_shape=MNI_SHAPE,
        order=order,
        mode="constant",
        cval=0.0,
    )
    return mni_data, MNI_AFFINE.copy()
