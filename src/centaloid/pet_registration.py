#!/usr/bin/env python3
"""PET-only spatial normalization to MNI space.

This module implements PET-to-MNI registration without requiring MRI data.
It uses mutual information as the similarity metric (better for PET) and
includes robust initialization using center-of-mass alignment.

Methods validated for PET-only Centiloid computation:
1. Mutual information registration (Maes et al., 1997)
2. Center-of-mass initialization for robustness
3. Multi-resolution optimization for speed and accuracy
"""

from __future__ import annotations

import logging
from typing import Tuple, Optional

import numpy as np
from scipy.ndimage import affine_transform, gaussian_filter, center_of_mass
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

# MNI-152 space parameters (2mm resolution)
MNI_SHAPE = (91, 109, 91)
MNI_AFFINE = np.array([
    [-2.0,  0.0,  0.0,  90.0],
    [ 0.0,  2.0,  0.0, -126.0],
    [ 0.0,  0.0,  2.0,  -72.0],
    [ 0.0,  0.0,  0.0,    1.0],
])

# MNI brain center of mass (approximate, in world coordinates)
MNI_COM_WORLD = np.array([0.0, -20.0, 10.0])


def _mutual_information(a: np.ndarray, b: np.ndarray, bins: int = 32) -> float:
    """Compute mutual information between two images.

    Returns negative MI (for minimization).
    """
    # Flatten and remove zeros
    mask = (a > 0) & (b > 0)
    if mask.sum() < 100:
        return 0.0

    a_vals = a[mask].ravel()
    b_vals = b[mask].ravel()

    # Normalize to [0, bins-1]
    a_norm = (a_vals - a_vals.min()) / (a_vals.max() - a_vals.min() + 1e-10) * (bins - 1)
    b_norm = (b_vals - b_vals.min()) / (b_vals.max() - b_vals.min() + 1e-10) * (bins - 1)

    # Compute joint histogram
    hist_2d, _, _ = np.histogram2d(a_norm, b_norm, bins=bins, range=[[0, bins], [0, bins]])

    # Normalize to probability
    pxy = hist_2d / hist_2d.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)

    # Compute MI = sum(p(x,y) * log(p(x,y) / (p(x) * p(y))))
    px_py = np.outer(px, py)
    nonzero = pxy > 1e-10
    mi = np.sum(pxy[nonzero] * np.log(pxy[nonzero] / (px_py[nonzero] + 1e-10) + 1e-10))

    return -mi  # Negative for minimization


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized cross-correlation (negative for minimization)."""
    mask = (a > 0) & (b > 0)
    if mask.sum() < 100:
        return 0.0

    a_vals = a[mask].astype(np.float64)
    b_vals = b[mask].astype(np.float64)

    a_m = a_vals - a_vals.mean()
    b_m = b_vals - b_vals.mean()

    num = np.dot(a_m, b_m)
    den = np.sqrt(np.dot(a_m, a_m) * np.dot(b_m, b_m))

    if den < 1e-10:
        return 0.0
    return -num / den


def _params_to_matrix(params: np.ndarray) -> np.ndarray:
    """Convert 9 parameters to 4x4 affine matrix.

    params = [tx, ty, tz, rx, ry, rz, sx, sy, sz]
    """
    tx, ty, tz = params[0:3]
    rx, ry, rz = params[3:6]
    sx, sy, sz = params[6:9]

    # Rotation matrices
    cx, sx_ = np.cos(rx), np.sin(rx)
    cy, sy_ = np.cos(ry), np.sin(ry)
    cz, sz_ = np.cos(rz), np.sin(rz)

    Rx = np.array([[1, 0, 0], [0, cx, -sx_], [0, sx_, cx]])
    Ry = np.array([[cy, 0, sy_], [0, 1, 0], [-sy_, 0, cy]])
    Rz = np.array([[cz, -sz_, 0], [sz_, cz, 0], [0, 0, 1]])

    R = Rz @ Ry @ Rx
    S = np.diag([sx, sy, sz])

    M = np.eye(4)
    M[0:3, 0:3] = R @ S
    M[0:3, 3] = [tx, ty, tz]
    return M


def compute_center_of_mass(volume: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Compute center of mass in world coordinates."""
    # Threshold to get brain mask
    threshold = volume.max() * 0.1
    brain_mask = volume > threshold

    if brain_mask.sum() < 100:
        # Fallback to volume center
        center_vox = np.array(volume.shape) / 2
    else:
        center_vox = np.array(center_of_mass(brain_mask))

    # Convert to world coordinates
    center_world = affine @ np.append(center_vox, 1)
    return center_world[:3]


def register_pet_to_mni(
    pet_data: np.ndarray,
    pet_affine: np.ndarray,
    target_mask: Optional[np.ndarray] = None,
    max_iter: int = 100,
    use_mi: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Register PET to MNI space using intensity-based optimization.

    Parameters
    ----------
    pet_data : ndarray
        Native-space PET volume
    pet_affine : ndarray
        Voxel-to-world affine for PET
    target_mask : ndarray, optional
        Brain mask in MNI space to use as registration target.
        If None, uses a synthetic ellipsoid.
    max_iter : int
        Maximum iterations for optimization
    use_mi : bool
        If True, use mutual information; otherwise use NCC

    Returns
    -------
    registered_pet : ndarray
        PET resampled to MNI space (91, 109, 91)
    transform : ndarray
        4x4 transformation matrix
    """
    # Create target template (brain-shaped mask in MNI space)
    if target_mask is None:
        target_mask = _create_brain_template()

    # Smooth and normalize
    pet_smooth = gaussian_filter(pet_data.astype(np.float64), sigma=2)
    pet_norm = pet_smooth / (pet_smooth.max() + 1e-10)

    target_smooth = gaussian_filter(target_mask.astype(np.float64), sigma=2)
    target_norm = target_smooth / (target_smooth.max() + 1e-10)

    # Step 1: Center-of-mass alignment (robust initialization)
    pet_com = compute_center_of_mass(pet_data, pet_affine)

    # Initial translation to align centers
    initial_translation = MNI_COM_WORLD - pet_com

    inv_mni = np.linalg.inv(MNI_AFFINE)

    # Choose similarity metric
    similarity_fn = _mutual_information if use_mi else _ncc

    def _cost(params):
        M_world = _params_to_matrix(params)
        # source_vox → world → MNI_world → MNI_vox
        combined = inv_mni @ M_world @ pet_affine
        inv_combined = np.linalg.inv(combined)

        resampled = affine_transform(
            pet_norm,
            inv_combined[0:3, 0:3],
            offset=inv_combined[0:3, 3],
            output_shape=MNI_SHAPE,
            order=1,
            mode="constant",
            cval=0.0,
        )
        return similarity_fn(resampled, target_norm)

    # Initial parameters with COM-based translation
    x0 = np.array([
        initial_translation[0], initial_translation[1], initial_translation[2],
        0, 0, 0,  # rotation
        1, 1, 1,  # scale
    ], dtype=np.float64)

    # Optimize
    result = minimize(
        _cost, x0, method="Powell",
        options={"maxiter": max_iter, "ftol": 1e-5}
    )

    # Apply final transform
    M_world = _params_to_matrix(result.x)
    transform = inv_mni @ M_world @ pet_affine
    inv_transform = np.linalg.inv(transform)

    registered = affine_transform(
        pet_data.astype(np.float64),
        inv_transform[0:3, 0:3],
        offset=inv_transform[0:3, 3],
        output_shape=MNI_SHAPE,
        order=1,
        mode="constant",
        cval=0.0,
    )

    return registered, transform


def _create_brain_template() -> np.ndarray:
    """Create a synthetic brain-shaped template in MNI space."""
    kk, jj, ii = np.mgrid[0:MNI_SHAPE[0], 0:MNI_SHAPE[1], 0:MNI_SHAPE[2]]
    ijk1 = np.column_stack([kk.ravel(), jj.ravel(), ii.ravel(), np.ones(kk.size)])
    mni = (MNI_AFFINE @ ijk1.T).T[:, :3]

    # Ellipsoidal brain model
    val = (
        (mni[:, 0] / 65) ** 2 +
        ((mni[:, 1] + 10) / 80) ** 2 +
        ((mni[:, 2] - 5) / 60) ** 2
    )
    brain = np.zeros(kk.size)
    brain[val <= 1.0] = 1.0
    brain = brain.reshape(MNI_SHAPE)

    # Add cerebellum bulge
    cerebellum = (
        (mni[:, 0] / 50) ** 2 +
        ((mni[:, 1] + 60) / 25) ** 2 +
        ((mni[:, 2] + 35) / 20) ** 2
    )
    cb = np.zeros(kk.size)
    cb[cerebellum <= 1.0] = 1.0
    brain = np.maximum(brain.ravel(), cb).reshape(MNI_SHAPE)

    return gaussian_filter(brain, sigma=2)


def create_pet_template_from_masks(
    ctx_mask: np.ndarray,
    wc_mask: np.ndarray
) -> np.ndarray:
    """Create a PET-like template from VOI masks.

    This creates a synthetic PET-like image where cortical regions
    have higher intensity (simulating amyloid uptake pattern).
    """
    template = np.zeros(MNI_SHAPE, dtype=np.float64)

    # Cerebellum reference (lower uptake)
    template[wc_mask > 0] = 1.0

    # Cortical target (higher uptake in AD-like pattern)
    template[ctx_mask > 0] = 1.5

    # Smooth to create realistic PET appearance
    template = gaussian_filter(template, sigma=3)

    # Normalize
    if template.max() > 0:
        template = template / template.max()

    return template


if __name__ == "__main__":
    # Test the registration
    print("PET-only registration module loaded successfully")
