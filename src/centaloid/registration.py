"""Spatial normalisation of PET volumes to MNI-152 space via ANTs.

Primary method uses ``antsRegistrationSyNQuick.sh`` (must be on ``$PATH``).
Falls back to a lightweight scipy-based affine if ANTs is unavailable.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from typing import Optional

import numpy as np
from scipy.ndimage import affine_transform, gaussian_filter
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MNI-152 template stub (2 mm resolution, 91x109x91)
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


# ---------------------------------------------------------------------------
# NIfTI I/O helpers (uses nibabel when available)
# ---------------------------------------------------------------------------

def _save_nifti(data: np.ndarray, affine: np.ndarray, path: str) -> None:
    """Save a volume as a NIfTI-1 file."""
    import nibabel as nib
    img = nib.Nifti1Image(data.astype(np.float32), affine)
    nib.save(img, path)


def _load_nifti(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a NIfTI file and return (data, affine)."""
    import nibabel as nib
    img = nib.load(path)
    return np.asarray(img.dataobj, dtype=np.float64), img.affine.copy()


# ---------------------------------------------------------------------------
# Synthetic MNI template (used as the fixed image for registration)
# ---------------------------------------------------------------------------

def _synthetic_mni_template() -> np.ndarray:
    """Create an approximate MNI brain-shaped template for registration.

    Produces a smooth ellipsoidal "brain" in MNI space that is sufficient
    for gross alignment.  If a real MNI-152 PET template is available on
    disk it should be preferred.
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
# ANTs-based registration (preferred)
# ---------------------------------------------------------------------------

def _ants_available() -> bool:
    """Return True if antsRegistrationSyNQuick.sh is on PATH."""
    return shutil.which("antsRegistrationSyNQuick.sh") is not None


def _register_with_ants(
    source_data: np.ndarray,
    source_affine: np.ndarray,
    fixed_path: Optional[str] = None,
    transform_type: str = "a",
) -> tuple[np.ndarray, np.ndarray]:
    """Register *source_data* to MNI space using ANTs.

    Parameters
    ----------
    source_data : (Z, Y, X) ndarray
    source_affine : (4, 4) voxel-to-world
    fixed_path : path to the fixed template NIfTI.  When ``None`` the
        synthetic MNI template is written to a temp file.
    transform_type : str
        ANTs transform code passed to ``-t``.
        ``"a"`` = rigid + affine, ``"s"`` = rigid + affine + SyN.

    Returns
    -------
    (warped_data, mni_affine) – the PET volume resampled in MNI space.
    """
    tmpdir = tempfile.mkdtemp(prefix="centaloid_ants_")
    try:
        # --- write the moving image ---
        moving_path = os.path.join(tmpdir, "moving.nii.gz")
        _save_nifti(source_data, source_affine, moving_path)

        # --- write the fixed (template) image ---
        if fixed_path is None:
            fixed_path = os.path.join(tmpdir, "fixed.nii.gz")
            _save_nifti(_synthetic_mni_template(), MNI_AFFINE, fixed_path)

        out_prefix = os.path.join(tmpdir, "reg_")

        cmd = [
            "antsRegistrationSyNQuick.sh",
            "-d", "3",
            "-f", fixed_path,
            "-m", moving_path,
            "-t", transform_type,
            "-o", out_prefix,
            "-n", str(max(1, os.cpu_count() or 1)),
        ]
        logger.info("Running ANTs: %s", " ".join(cmd))

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )

        if proc.returncode != 0:
            logger.error("ANTs stderr:\n%s", proc.stderr)
            raise RuntimeError(
                f"antsRegistrationSyNQuick.sh exited with code {proc.returncode}"
            )

        # --- read the warped image ---
        warped_path = out_prefix + "Warped.nii.gz"
        if not os.path.isfile(warped_path):
            raise FileNotFoundError(
                f"Expected ANTs output not found: {warped_path}"
            )

        warped_data, warped_affine = _load_nifti(warped_path)
        logger.info("ANTs registration complete.")
        return warped_data, warped_affine

    finally:
        # Clean up temp files
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Scipy fallback registration (lightweight affine)
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
    """Convert 12 parameters to a 4x4 affine matrix.

    params = [tx, ty, tz, rx, ry, rz, sx, sy, sz, shxy, shxz, shyz]
    """
    tx, ty, tz = params[0:3]
    rx, ry, rz = params[3:6]
    sx, sy, sz = params[6:9]
    shxy, shxz, shyz = params[9:12]

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


def _estimate_affine_scipy(
    source_data: np.ndarray,
    source_affine: np.ndarray,
    max_iter: int = 200,
) -> np.ndarray:
    """Estimate a 4x4 source-voxel-to-MNI-voxel affine using scipy."""
    template = _synthetic_mni_template()
    source_smooth = gaussian_filter(source_data.astype(np.float64), sigma=2)

    s_max = source_smooth.max() or 1.0
    t_max = template.max() or 1.0
    src_norm = source_smooth / s_max
    tpl_norm = template / t_max

    inv_mni = np.linalg.inv(MNI_AFFINE)

    def _cost(params):
        M_world = _params_to_affine(params)
        combined = inv_mni @ M_world @ source_affine
        inv_combined = np.linalg.inv(combined)
        resampled = affine_transform(
            src_norm, inv_combined[0:3, 0:3], offset=inv_combined[0:3, 3],
            output_shape=MNI_SHAPE, order=1, mode="constant", cval=0.0,
        )
        return _ncc(resampled, tpl_norm)

    x0 = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=np.float64)

    logger.info("Starting scipy affine registration (%d max iter)…", max_iter)
    result = minimize(_cost, x0, method="Powell",
                      options={"maxiter": max_iter, "ftol": 1e-6})
    logger.info("Scipy registration done: NCC=%.4f, nfev=%d",
                -result.fun, result.nfev)

    M_world = _params_to_affine(result.x)
    return inv_mni @ M_world @ source_affine


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate_affine_to_mni(
    source_data: np.ndarray,
    source_affine: np.ndarray,
    max_iter: int = 200,
) -> np.ndarray:
    """Estimate a 4x4 affine mapping source voxels to MNI voxels.

    Uses ``antsRegistrationSyNQuick.sh`` when available on ``$PATH``,
    otherwise falls back to scipy-based NCC optimisation.

    Returns
    -------
    transform : (4, 4) ndarray
        Maps **source voxel** indices to **MNI voxel** indices.
    """
    if _ants_available():
        logger.info("ANTs detected – using antsRegistrationSyNQuick.sh")
        # ANTs performs registration + resampling in one shot; we store the
        # result and return an identity transform so that resample_to_mni
        # can short-circuit when it detects the ANTs cache.
        warped, _ = _register_with_ants(source_data, source_affine)
        # Stash the warped volume so resample_to_mni can retrieve it.
        _ants_cache["warped"] = warped
        return np.eye(4)
    else:
        logger.info("ANTs not found – falling back to scipy affine registration")
        return _estimate_affine_scipy(source_data, source_affine, max_iter)


# Module-level cache for ANTs warped output (avoids resampling twice)
_ants_cache: dict = {}


def resample_to_mni(
    source_data: np.ndarray,
    source_affine: np.ndarray,
    transform: Optional[np.ndarray] = None,
    order: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample *source_data* into MNI-152 space (2 mm).

    If ANTs was used for registration the already-warped volume is returned
    directly, avoiding double resampling.

    Parameters
    ----------
    source_data : (Z, Y, X) ndarray
    source_affine : (4, 4) voxel-to-world
    transform : (4, 4) source-voxel-to-MNI-voxel (if None, estimated).
    order : int  Spline interpolation order.

    Returns
    -------
    (mni_data, MNI_AFFINE)
    """
    # If ANTs already produced a warped volume, return it directly
    if "warped" in _ants_cache:
        warped = _ants_cache.pop("warped")
        return warped, MNI_AFFINE.copy()

    if transform is None:
        transform = estimate_affine_to_mni(source_data, source_affine)
        # Check again in case ANTs was used
        if "warped" in _ants_cache:
            warped = _ants_cache.pop("warped")
            return warped, MNI_AFFINE.copy()

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
