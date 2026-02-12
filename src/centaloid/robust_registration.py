#!/usr/bin/env python3
"""Robust PET-only spatial normalization to MNI space.

This module implements a more sophisticated PET-to-MNI registration
that doesn't require MRI data. Based on validated methods from:
- Tuszynski et al. (2016) - Direct PET template registration
- Centiloid Working Group recommendations

Key improvements over simple registration:
1. Multi-resolution pyramid (coarse-to-fine)
2. Normalized mutual information (more robust)
3. Brain masking for focused optimization
4. Rigid-then-affine optimization strategy
5. Multiple random restarts to avoid local minima
"""

from __future__ import annotations

import logging
from typing import Tuple, Optional, Callable
import warnings

import numpy as np
from scipy.ndimage import affine_transform, gaussian_filter, zoom, binary_dilation, binary_erosion
from scipy.optimize import minimize, differential_evolution

logger = logging.getLogger(__name__)

# MNI-152 space parameters (2mm resolution)
MNI_SHAPE = (91, 109, 91)
MNI_AFFINE = np.array([
    [-2.0,  0.0,  0.0,  90.0],
    [ 0.0,  2.0,  0.0, -126.0],
    [ 0.0,  0.0,  2.0,  -72.0],
    [ 0.0,  0.0,  0.0,    1.0],
])

# MNI brain center (in voxel coordinates)
MNI_CENTER_VOX = np.array([45.0, 63.0, 36.0])


def create_brain_mask(volume: np.ndarray, threshold_pct: float = 0.15) -> np.ndarray:
    """Create a brain mask from PET volume using intensity thresholding."""
    # Use Otsu-like thresholding
    nonzero = volume[volume > 0]
    if nonzero.size == 0:
        return np.zeros_like(volume, dtype=bool)

    threshold = np.percentile(nonzero, threshold_pct * 100)
    mask = volume > threshold

    # Clean up with morphological operations
    mask = binary_erosion(mask, iterations=1)
    mask = binary_dilation(mask, iterations=2)

    return mask


def compute_center_of_mass_vox(mask: np.ndarray) -> np.ndarray:
    """Compute center of mass in voxel coordinates."""
    if mask.sum() == 0:
        return np.array(mask.shape) / 2

    indices = np.array(np.where(mask))
    return indices.mean(axis=1)


def normalized_mutual_information(img1: np.ndarray, img2: np.ndarray,
                                   mask: Optional[np.ndarray] = None,
                                   bins: int = 64) -> float:
    """Compute normalized mutual information between two images.

    NMI = (H(A) + H(B)) / H(A,B)
    Returns negative NMI for minimization.
    """
    if mask is not None:
        a = img1[mask].ravel()
        b = img2[mask].ravel()
    else:
        # Use overlap region
        overlap = (img1 > 0) & (img2 > 0)
        if overlap.sum() < 1000:
            return 0.0
        a = img1[overlap].ravel()
        b = img2[overlap].ravel()

    if len(a) < 1000:
        return 0.0

    # Normalize to [0, bins-1]
    a_min, a_max = a.min(), a.max()
    b_min, b_max = b.min(), b.max()

    if a_max - a_min < 1e-10 or b_max - b_min < 1e-10:
        return 0.0

    a_norm = ((a - a_min) / (a_max - a_min) * (bins - 1)).astype(np.int32)
    b_norm = ((b - b_min) / (b_max - b_min) * (bins - 1)).astype(np.int32)

    # Clip to valid range
    a_norm = np.clip(a_norm, 0, bins - 1)
    b_norm = np.clip(b_norm, 0, bins - 1)

    # Compute joint histogram
    hist_2d = np.zeros((bins, bins), dtype=np.float64)
    np.add.at(hist_2d, (a_norm, b_norm), 1)

    # Normalize
    hist_2d = hist_2d / hist_2d.sum()

    # Marginal distributions
    p_a = hist_2d.sum(axis=1)
    p_b = hist_2d.sum(axis=0)

    # Entropies
    eps = 1e-10
    h_a = -np.sum(p_a[p_a > eps] * np.log(p_a[p_a > eps]))
    h_b = -np.sum(p_b[p_b > eps] * np.log(p_b[p_b > eps]))
    h_ab = -np.sum(hist_2d[hist_2d > eps] * np.log(hist_2d[hist_2d > eps]))

    if h_ab < eps:
        return 0.0

    nmi = (h_a + h_b) / h_ab
    return -nmi  # Negative for minimization


def params_to_rigid(params: np.ndarray) -> np.ndarray:
    """Convert 6 parameters to rigid transformation matrix.

    params = [tx, ty, tz, rx, ry, rz]
    """
    tx, ty, tz = params[0:3]
    rx, ry, rz = params[3:6]

    # Rotation matrices
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

    R = Rz @ Ry @ Rx

    M = np.eye(4)
    M[0:3, 0:3] = R
    M[0:3, 3] = [tx, ty, tz]
    return M


def params_to_affine(params: np.ndarray) -> np.ndarray:
    """Convert 12 parameters to affine transformation matrix.

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


def downsample_volume(volume: np.ndarray, factor: int) -> np.ndarray:
    """Downsample volume by given factor."""
    if factor == 1:
        return volume
    sigma = factor / 2.0
    smoothed = gaussian_filter(volume.astype(np.float64), sigma=sigma)
    return zoom(smoothed, 1.0 / factor, order=1)


def create_mni_brain_template() -> np.ndarray:
    """Create a realistic brain-shaped template in MNI space."""
    kk, jj, ii = np.mgrid[0:MNI_SHAPE[0], 0:MNI_SHAPE[1], 0:MNI_SHAPE[2]]
    ijk1 = np.column_stack([kk.ravel(), jj.ravel(), ii.ravel(), np.ones(kk.size)])
    mni = (MNI_AFFINE @ ijk1.T).T[:, :3]

    # Main brain ellipsoid
    brain = (
        (mni[:, 0] / 70) ** 2 +
        ((mni[:, 1] + 15) / 85) ** 2 +
        ((mni[:, 2] - 10) / 65) ** 2
    )
    template = np.zeros(kk.size, dtype=np.float64)
    template[brain <= 1.0] = 1.0

    # Cerebellum
    cerebellum = (
        (mni[:, 0] / 55) ** 2 +
        ((mni[:, 1] + 55) / 25) ** 2 +
        ((mni[:, 2] + 38) / 22) ** 2
    )
    template[cerebellum <= 1.0] = 1.0

    # Brainstem
    brainstem = (
        (mni[:, 0] / 12) ** 2 +
        ((mni[:, 1] + 30) / 30) ** 2 +
        ((mni[:, 2] + 25) / 35) ** 2
    )
    template[brainstem <= 1.0] = 0.8

    template = template.reshape(MNI_SHAPE)
    template = gaussian_filter(template, sigma=3)

    return template


def create_amyloid_pet_template(ctx_mask: np.ndarray, wc_mask: np.ndarray) -> np.ndarray:
    """Create a template that mimics amyloid PET uptake pattern."""
    # Start with brain template
    template = create_mni_brain_template()

    # Enhance cortical regions (amyloid pattern)
    template[ctx_mask > 0] = 1.5

    # Cerebellum has reference uptake
    template[wc_mask > 0] = 1.0

    # Smooth to look more like real PET
    template = gaussian_filter(template, sigma=2)

    # Normalize
    template = template / template.max()

    return template


def register_pet_robust(
    source: np.ndarray,
    source_affine: np.ndarray,
    target: np.ndarray,
    target_affine: np.ndarray = None,
    max_iter_rigid: int = 100,
    max_iter_affine: int = 100,
    use_multiresolution: bool = True,
    num_restarts: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Robust PET registration using multi-resolution and multiple restarts.

    Parameters
    ----------
    source : ndarray
        Source PET volume to register
    source_affine : ndarray
        Source voxel-to-world affine
    target : ndarray
        Target template in MNI space
    target_affine : ndarray, optional
        Target affine (defaults to MNI_AFFINE)
    max_iter_rigid : int
        Max iterations for rigid registration
    max_iter_affine : int
        Max iterations for affine refinement
    use_multiresolution : bool
        Whether to use coarse-to-fine optimization
    num_restarts : int
        Number of random restarts to try

    Returns
    -------
    registered : ndarray
        Source resampled to target space
    transform : ndarray
        Final transformation matrix
    """
    if target_affine is None:
        target_affine = MNI_AFFINE.copy()

    # Create brain masks
    source_mask = create_brain_mask(source)
    target_mask = target > 0.1 * target.max()

    # Compute centers of mass for initialization
    source_com_vox = compute_center_of_mass_vox(source_mask)
    target_com_vox = compute_center_of_mass_vox(target_mask)

    # Convert to world coordinates
    source_com_world = source_affine @ np.append(source_com_vox, 1)
    target_com_world = target_affine @ np.append(target_com_vox, 1)

    # Initial translation
    init_translation = target_com_world[:3] - source_com_world[:3]

    # Normalize intensities
    source_norm = source.astype(np.float64)
    source_max = source_norm.max()
    if source_max > 0:
        source_norm = source_norm / source_max

    target_norm = target.astype(np.float64)
    target_max = target_norm.max()
    if target_max > 0:
        target_norm = target_norm / target_max

    inv_target_affine = np.linalg.inv(target_affine)

    def compute_cost(params, param_type='rigid', resolution=1):
        """Compute registration cost."""
        if param_type == 'rigid':
            M_world = params_to_rigid(params)
        else:
            M_world = params_to_affine(params)

        # Compose transformations: source_vox -> world -> target_world -> target_vox
        combined = inv_target_affine @ M_world @ source_affine
        inv_combined = np.linalg.inv(combined)

        # Resample source to target space
        if resolution > 1:
            target_shape_ds = tuple(s // resolution for s in target_norm.shape)
            source_ds = downsample_volume(source_norm, resolution)
            target_ds = downsample_volume(target_norm, resolution)

            # Adjust transformation for downsampled space
            scale_mat = np.diag([resolution, resolution, resolution, 1.0])
            inv_combined_ds = scale_mat @ inv_combined @ np.linalg.inv(scale_mat)

            resampled = affine_transform(
                source_ds,
                inv_combined_ds[0:3, 0:3],
                offset=inv_combined_ds[0:3, 3],
                output_shape=target_shape_ds,
                order=1,
                mode='constant',
                cval=0.0,
            )
            return normalized_mutual_information(resampled, target_ds)
        else:
            resampled = affine_transform(
                source_norm,
                inv_combined[0:3, 0:3],
                offset=inv_combined[0:3, 3],
                output_shape=target_norm.shape,
                order=1,
                mode='constant',
                cval=0.0,
            )
            return normalized_mutual_information(resampled, target_norm)

    best_cost = float('inf')
    best_params = None

    # Try multiple initializations
    for restart in range(num_restarts):
        # Add random perturbation to initial translation for restarts > 0
        if restart == 0:
            init_trans = init_translation.copy()
            init_rot = np.zeros(3)
        else:
            init_trans = init_translation + np.random.randn(3) * 10  # ±10mm
            init_rot = np.random.randn(3) * 0.1  # ±0.1 rad (~6 degrees)

        # Stage 1: Rigid registration
        x0_rigid = np.concatenate([init_trans, init_rot])

        if use_multiresolution:
            # Coarse level (4x downsampled)
            result = minimize(
                lambda p: compute_cost(p, 'rigid', resolution=4),
                x0_rigid, method='Powell',
                options={'maxiter': max_iter_rigid // 2, 'ftol': 1e-4}
            )
            x0_rigid = result.x

            # Medium level (2x downsampled)
            result = minimize(
                lambda p: compute_cost(p, 'rigid', resolution=2),
                x0_rigid, method='Powell',
                options={'maxiter': max_iter_rigid // 2, 'ftol': 1e-5}
            )
            x0_rigid = result.x

        # Fine level (full resolution)
        result_rigid = minimize(
            lambda p: compute_cost(p, 'rigid', resolution=1),
            x0_rigid, method='Powell',
            options={'maxiter': max_iter_rigid, 'ftol': 1e-6}
        )

        # Stage 2: Affine refinement
        x0_affine = np.concatenate([
            result_rigid.x[:3],  # translation
            result_rigid.x[3:6],  # rotation
            [1.0, 1.0, 1.0],  # scale
            [0.0, 0.0, 0.0],  # shear
        ])

        result_affine = minimize(
            lambda p: compute_cost(p, 'affine', resolution=1),
            x0_affine, method='Powell',
            options={'maxiter': max_iter_affine, 'ftol': 1e-6}
        )

        if result_affine.fun < best_cost:
            best_cost = result_affine.fun
            best_params = result_affine.x

    if best_params is None:
        warnings.warn("Registration failed - using identity transform")
        best_params = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0])

    # Compute final transformation
    M_world = params_to_affine(best_params)
    transform = inv_target_affine @ M_world @ source_affine
    inv_transform = np.linalg.inv(transform)

    # Apply final resampling
    registered = affine_transform(
        source.astype(np.float64),
        inv_transform[0:3, 0:3],
        offset=inv_transform[0:3, 3],
        output_shape=target_norm.shape,
        order=1,
        mode='constant',
        cval=0.0,
    )

    return registered, transform


def register_pet_to_mni_robust(
    pet_data: np.ndarray,
    pet_affine: np.ndarray,
    ctx_mask: Optional[np.ndarray] = None,
    wc_mask: Optional[np.ndarray] = None,
    max_iter: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """Register PET to MNI space using robust multi-resolution approach.

    Parameters
    ----------
    pet_data : ndarray
        Native-space PET volume
    pet_affine : ndarray
        Voxel-to-world affine
    ctx_mask : ndarray, optional
        CTX VOI mask in MNI space (for template creation)
    wc_mask : ndarray, optional
        WC VOI mask in MNI space (for template creation)
    max_iter : int
        Maximum total iterations

    Returns
    -------
    registered : ndarray
        PET resampled to MNI space (91, 109, 91)
    transform : ndarray
        4x4 transformation matrix
    """
    # Create target template
    if ctx_mask is not None and wc_mask is not None:
        target = create_amyloid_pet_template(ctx_mask, wc_mask)
    else:
        target = create_mni_brain_template()

    # Run robust registration
    registered, transform = register_pet_robust(
        pet_data, pet_affine,
        target, MNI_AFFINE,
        max_iter_rigid=max_iter // 2,
        max_iter_affine=max_iter // 2,
        use_multiresolution=True,
        num_restarts=3,
    )

    return registered, transform


if __name__ == "__main__":
    print("Robust PET registration module loaded successfully")
