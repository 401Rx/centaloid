#!/usr/bin/env python3
"""Validate the Centiloid pipeline against GAAIN PiB calibration data.

This script processes the official GAAIN PiB calibration dataset (45 AD + 34 YC subjects)
and compares the computed Centiloid values against the expected values from the
Centiloid project's SupplementaryTable1.xlsx.

The GAAIN PiB data is already registered to MNI-152 2mm space, so no registration
is required - we simply apply the VOI masks and calculate SUVr/Centiloid.

Usage:
    python validate_gaain_pib.py /path/to/gaain/data --output results.csv

Expected directory structure:
    /path/to/gaain/data/
        AD01_PiB_5070.nii
        AD02_PiB_5070.nii
        ...
        AD45_PiB_5070.nii
        YC101_PiB_5070.nii
        ...
        YC134_PiB_5070.nii
        voi_ctx_2mm.nii         (or voi_Ctx_2mm.nii)
        voi_WhlCbl_2mm.nii      (whole cerebellum)
        SupplementaryTable1.xlsx (optional, for comparison)

References:
    Klunk WE et al. Alzheimers Dement 2015;11(1):1-15.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import warnings

import numpy as np

try:
    import nibabel as nib
except ImportError:
    print("ERROR: nibabel is required. Install with: pip install nibabel")
    sys.exit(1)

try:
    from scipy.ndimage import affine_transform
except ImportError:
    print("ERROR: scipy is required. Install with: pip install scipy")
    sys.exit(1)

# Add parent path for centaloid imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from centaloid.robust_registration import (
        register_pet_to_mni_robust,
        create_amyloid_pet_template,
        MNI_AFFINE,
        MNI_SHAPE,
    )
    HAS_ROBUST_REGISTRATION = True
except ImportError:
    HAS_ROBUST_REGISTRATION = False

try:
    from centaloid.pet_registration import (
        register_pet_to_mni,
        create_pet_template_from_masks,
    )
    HAS_PET_REGISTRATION = True
except ImportError:
    HAS_PET_REGISTRATION = False

try:
    from centaloid.registration import resample_to_mni
    HAS_REGISTRATION = True
except ImportError:
    HAS_REGISTRATION = False

if not HAS_ROBUST_REGISTRATION and not HAS_PET_REGISTRATION and not HAS_REGISTRATION:
    print("WARNING: No registration module available - using simple resampling")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("WARNING: pandas not installed - Excel comparison will be skipped")


# ---------------------------------------------------------------------------
# PiB Centiloid conversion parameters (anchor tracer)
# ---------------------------------------------------------------------------

# From Klunk et al. 2015, Table 1
# For PiB: CL = 100 * (SUVr - SUVr_YC) / (SUVr_AD - SUVr_YC)
# Where SUVr_YC = 1.009, SUVr_AD = 2.068
# This simplifies to: CL = 94.6 * SUVr - 95.4
# But the published linear equation is: CL = 94.6 * SUVr - 94.6
PIB_SLOPE = 94.6
PIB_INTERCEPT = -94.6


@dataclass
class SubjectResult:
    """Result for a single subject."""
    subject_id: str
    group: str  # "AD" or "YC"
    ctx_mean: float
    wc_mean: float
    suvr: float
    centiloid: float
    expected_suvr: Optional[float] = None
    expected_centiloid: Optional[float] = None


def load_nifti(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a NIfTI file and return (data, affine)."""
    img = nib.load(path)
    data = np.asarray(img.dataobj, dtype=np.float32)
    # Handle 4D volumes (squeeze last dimension if size 1)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[:, :, :, 0]
    return data, img.affine


def resample_to_target(
    source_data: np.ndarray,
    source_affine: np.ndarray,
    target_shape: tuple,
    target_affine: np.ndarray,
) -> np.ndarray:
    """Resample source volume to target space using affine transformation."""
    from scipy.ndimage import affine_transform

    # Compute transformation from target voxels to source voxels
    # target_vox -> world -> source_vox
    target_to_world = target_affine
    world_to_source = np.linalg.inv(source_affine)
    target_to_source = world_to_source @ target_to_world

    # Extract rotation/scale matrix and offset
    matrix = target_to_source[:3, :3]
    offset = target_to_source[:3, 3]

    # Resample using trilinear interpolation
    resampled = affine_transform(
        source_data,
        matrix,
        offset=offset,
        output_shape=target_shape,
        order=1,  # trilinear
        mode='constant',
        cval=0.0,
    )
    return resampled


def compute_roi_mean(volume: np.ndarray, mask: np.ndarray) -> float:
    """Compute mean of volume within mask, excluding zeros."""
    vals = volume[mask > 0]
    # Exclude near-zero values (potential edge effects)
    valid = vals[vals > 0.01 * vals.mean()] if vals.size > 0 and vals.mean() > 0 else vals
    return float(valid.mean()) if valid.size > 0 else 0.0


def find_voi_file(data_dir: Path, patterns: list[str]) -> Optional[Path]:
    """Find a VOI file matching one of the patterns."""
    for pattern in patterns:
        # Try exact match
        path = data_dir / pattern
        if path.exists():
            return path
        # Try case-insensitive
        for f in data_dir.iterdir():
            if f.name.lower() == pattern.lower():
                return f
    return None


def find_subject_files(data_dir: Path) -> tuple[list[Path], list[Path]]:
    """Find AD and YC subject PiB files."""
    ad_files = sorted(data_dir.glob("AD*_PiB_*.nii*"))
    yc_files = sorted(data_dir.glob("YC*_PiB_*.nii*"))
    return ad_files, yc_files


def load_expected_values(excel_path: Path) -> dict[str, tuple[float, float]]:
    """Load expected SUVr and Centiloid values from SupplementaryTable1.xlsx.

    Returns dict mapping subject_id -> (expected_suvr, expected_centiloid)
    """
    if not HAS_PANDAS:
        return {}

    try:
        # Try to read the Excel file
        df = pd.read_excel(excel_path)

        # The table typically has columns like: Subject, SUVr, Centiloid
        # Try to find the relevant columns
        expected = {}

        # Common column name patterns
        subject_cols = ['Subject', 'subject', 'ID', 'id', 'Subject ID']
        suvr_cols = ['SUVr', 'SUVR', 'suvr', 'SUVr (CTX/WC)', 'PiB SUVr']
        cl_cols = ['Centiloid', 'CL', 'centiloid', 'PiB CL', 'Centiloid Value']

        subject_col = None
        suvr_col = None
        cl_col = None

        for col in df.columns:
            col_lower = col.lower().strip()
            if subject_col is None and any(s.lower() in col_lower for s in subject_cols):
                subject_col = col
            if suvr_col is None and 'suvr' in col_lower:
                suvr_col = col
            if cl_col is None and ('centiloid' in col_lower or col_lower == 'cl'):
                cl_col = col

        if subject_col is None:
            # Try first column as subject ID
            subject_col = df.columns[0]

        print(f"  Found columns: Subject='{subject_col}', SUVr='{suvr_col}', CL='{cl_col}'")

        for _, row in df.iterrows():
            subj = str(row[subject_col]).strip()
            # Extract just the subject ID (e.g., "AD01" from "AD01_PiB_5070")
            if '_' in subj:
                subj = subj.split('_')[0]

            suvr_val = float(row[suvr_col]) if suvr_col and pd.notna(row[suvr_col]) else None
            cl_val = float(row[cl_col]) if cl_col and pd.notna(row[cl_col]) else None

            if suvr_val is not None or cl_val is not None:
                expected[subj] = (suvr_val, cl_val)

        return expected

    except Exception as e:
        print(f"  Warning: Could not read Excel file: {e}")
        return {}


def validate_pipeline(
    data_dir: Path,
    output_path: Optional[Path] = None,
) -> list[SubjectResult]:
    """Run validation on all subjects in data_dir."""

    print("=" * 70)
    print("GAAIN PiB Centiloid Validation")
    print("=" * 70)
    print(f"Data directory: {data_dir}")
    print()

    # Find VOI files
    ctx_patterns = ['voi_ctx_2mm.nii', 'voi_Ctx_2mm.nii', 'voi_ctx_2mm.nii.gz']
    wc_patterns = ['voi_WhlCbl_2mm.nii', 'voi_whlcbl_2mm.nii', 'voi_WC_2mm.nii',
                   'voi_wc_2mm.nii', 'voi_WhlCbl_2mm.nii.gz']

    ctx_path = find_voi_file(data_dir, ctx_patterns)
    wc_path = find_voi_file(data_dir, wc_patterns)

    if ctx_path is None:
        print("ERROR: Could not find CTX VOI file (voi_ctx_2mm.nii)")
        print(f"  Searched in: {data_dir}")
        print(f"  Looking for: {ctx_patterns}")
        sys.exit(1)

    if wc_path is None:
        print("ERROR: Could not find WC VOI file (voi_WhlCbl_2mm.nii)")
        print(f"  Searched in: {data_dir}")
        print(f"  Looking for: {wc_patterns}")
        sys.exit(1)

    print(f"CTX VOI: {ctx_path.name}")
    print(f"WC VOI:  {wc_path.name}")

    # Load VOI masks
    ctx_mask, ctx_affine = load_nifti(ctx_path)
    wc_mask, wc_affine = load_nifti(wc_path)

    ctx_voxels = int((ctx_mask > 0).sum())
    wc_voxels = int((wc_mask > 0).sum())
    print(f"CTX voxels: {ctx_voxels}")
    print(f"WC voxels:  {wc_voxels}")

    # Create PET template from VOI masks for registration
    pet_template = None
    if HAS_ROBUST_REGISTRATION:
        print("Creating amyloid PET template for registration...")
        pet_template = create_amyloid_pet_template(ctx_mask > 0, wc_mask > 0)
        print("  Registration method: Robust multi-resolution (NMI + restarts)")
    elif HAS_PET_REGISTRATION:
        print("Creating PET template from VOI masks for registration...")
        pet_template = create_pet_template_from_masks(ctx_mask > 0, wc_mask > 0)
        print("  Registration method: PET-specific (mutual information + COM init)")
    elif HAS_REGISTRATION:
        print("  Registration method: T1-based (may be less accurate for PET)")
    else:
        print("  Registration method: Affine only (requires MNI-aligned data)")
    print()

    # Find subject files
    ad_files, yc_files = find_subject_files(data_dir)
    print(f"Found {len(ad_files)} AD subjects, {len(yc_files)} YC subjects")

    if len(ad_files) == 0 and len(yc_files) == 0:
        print("ERROR: No subject files found!")
        print("  Expected files like: AD01_PiB_5070.nii, YC101_PiB_5070.nii")
        sys.exit(1)

    # Load expected values if available
    excel_path = data_dir / "SupplementaryTable1.xlsx"
    expected_values = {}
    if excel_path.exists():
        print(f"Loading expected values from: {excel_path.name}")
        expected_values = load_expected_values(excel_path)
        print(f"  Loaded {len(expected_values)} expected values")
    print()

    # Process all subjects
    results: list[SubjectResult] = []

    print("-" * 70)
    print(f"{'Subject':<12} {'CTX Mean':>10} {'WC Mean':>10} {'SUVr':>8} {'CL':>8} {'Exp CL':>8} {'Diff':>8}")
    print("-" * 70)

    all_files = [('AD', f) for f in ad_files] + [('YC', f) for f in yc_files]

    for group, filepath in all_files:
        # Extract subject ID
        subject_id = filepath.stem.split('_')[0]  # e.g., "AD01" from "AD01_PiB_5070"

        # Load PET volume
        try:
            pet_data, pet_affine = load_nifti(filepath)
        except Exception as e:
            print(f"  ERROR loading {filepath.name}: {e}")
            continue

        # Check dimensions match VOI - if not, register PET to MNI space
        if pet_data.shape != ctx_mask.shape:
            if HAS_ROBUST_REGISTRATION:
                print(f"  Registering {subject_id} (robust)...", end=" ", flush=True)
                pet_data, _ = register_pet_to_mni_robust(
                    pet_data, pet_affine,
                    ctx_mask=ctx_mask > 0,
                    wc_mask=wc_mask > 0,
                    max_iter=200,
                )
                print("done")
            elif HAS_PET_REGISTRATION:
                print(f"  Registering {subject_id} (PET-specific)...", end=" ", flush=True)
                pet_data, _ = register_pet_to_mni(
                    pet_data, pet_affine,
                    target_mask=pet_template,
                    max_iter=150,
                    use_mi=True
                )
                print("done")
            elif HAS_REGISTRATION:
                print(f"  Registering {subject_id} (T1-based)...", end=" ", flush=True)
                pet_data, _ = resample_to_mni(pet_data, pet_affine)
                print("done")
            else:
                print(f"  Resampling {subject_id} (affine only)...", end=" ", flush=True)
                pet_data = resample_to_target(
                    pet_data, pet_affine,
                    ctx_mask.shape, ctx_affine
                )
                print("done")

        # Compute ROI means
        ctx_mean = compute_roi_mean(pet_data, ctx_mask)
        wc_mean = compute_roi_mean(pet_data, wc_mask)

        # Compute SUVr and Centiloid
        if wc_mean > 0:
            suvr = ctx_mean / wc_mean
        else:
            suvr = 0.0

        centiloid = PIB_SLOPE * suvr + PIB_INTERCEPT

        # Get expected values
        exp_suvr, exp_cl = expected_values.get(subject_id, (None, None))

        # Create result
        result = SubjectResult(
            subject_id=subject_id,
            group=group,
            ctx_mean=ctx_mean,
            wc_mean=wc_mean,
            suvr=round(suvr, 4),
            centiloid=round(centiloid, 2),
            expected_suvr=exp_suvr,
            expected_centiloid=exp_cl,
        )
        results.append(result)

        # Print result
        diff_str = ""
        if exp_cl is not None:
            diff = centiloid - exp_cl
            diff_str = f"{diff:+.2f}"

        exp_cl_str = f"{exp_cl:.2f}" if exp_cl is not None else "N/A"

        print(f"{subject_id:<12} {ctx_mean:>10.4f} {wc_mean:>10.4f} {suvr:>8.4f} "
              f"{centiloid:>8.2f} {exp_cl_str:>8} {diff_str:>8}")

    print("-" * 70)
    print()

    # Compute statistics
    ad_results = [r for r in results if r.group == 'AD']
    yc_results = [r for r in results if r.group == 'YC']

    print("=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    if yc_results:
        yc_suvr = [r.suvr for r in yc_results]
        yc_cl = [r.centiloid for r in yc_results]
        print(f"Young Controls (n={len(yc_results)}):")
        print(f"  SUVr:      mean = {np.mean(yc_suvr):.4f}, SD = {np.std(yc_suvr):.4f}")
        print(f"  Centiloid: mean = {np.mean(yc_cl):.2f}, SD = {np.std(yc_cl):.2f}")
        print(f"             range = [{min(yc_cl):.2f}, {max(yc_cl):.2f}]")
        print()

    if ad_results:
        ad_suvr = [r.suvr for r in ad_results]
        ad_cl = [r.centiloid for r in ad_results]
        print(f"Alzheimer's Disease (n={len(ad_results)}):")
        print(f"  SUVr:      mean = {np.mean(ad_suvr):.4f}, SD = {np.std(ad_suvr):.4f}")
        print(f"  Centiloid: mean = {np.mean(ad_cl):.2f}, SD = {np.std(ad_cl):.2f}")
        print(f"             range = [{min(ad_cl):.2f}, {max(ad_cl):.2f}]")
        print()

    # Expected values from Klunk et al. 2015:
    # YC: SUVr = 1.009 (SD 0.038), CL = 0 by definition
    # AD: SUVr = 2.068 (SD 0.364), CL = 100 by definition
    print("Expected values (Klunk et al. 2015):")
    print("  YC: SUVr = 1.009 (SD 0.038), CL ~= 0")
    print("  AD: SUVr = 2.068 (SD 0.364), CL ~= 100")
    print()

    # Validation metrics
    if expected_values:
        computed_cl = []
        expected_cl = []
        for r in results:
            if r.expected_centiloid is not None:
                computed_cl.append(r.centiloid)
                expected_cl.append(r.expected_centiloid)

        if computed_cl:
            computed_cl = np.array(computed_cl)
            expected_cl = np.array(expected_cl)
            diff = computed_cl - expected_cl

            print("VALIDATION vs EXPECTED VALUES")
            print("-" * 70)
            print(f"  N comparisons: {len(diff)}")
            print(f"  Mean diff:     {np.mean(diff):.4f} CL")
            print(f"  SD diff:       {np.std(diff):.4f} CL")
            print(f"  Max abs diff:  {np.max(np.abs(diff)):.4f} CL")
            print(f"  Correlation:   {np.corrcoef(computed_cl, expected_cl)[0,1]:.6f}")

            # Check if within acceptable tolerance
            tolerance = 2.0  # CL units
            within_tol = np.sum(np.abs(diff) <= tolerance)
            print(f"  Within ±{tolerance} CL: {within_tol}/{len(diff)} ({100*within_tol/len(diff):.1f}%)")
            print()

    # Save results to CSV
    if output_path:
        print(f"Saving results to: {output_path}")
        with open(output_path, 'w') as f:
            f.write("Subject,Group,CTX_Mean,WC_Mean,SUVr,Centiloid,Expected_SUVr,Expected_CL,Diff_CL\n")
            for r in results:
                exp_suvr_str = f"{r.expected_suvr:.4f}" if r.expected_suvr else ""
                exp_cl_str = f"{r.expected_centiloid:.2f}" if r.expected_centiloid else ""
                diff_str = f"{r.centiloid - r.expected_centiloid:.2f}" if r.expected_centiloid else ""
                f.write(f"{r.subject_id},{r.group},{r.ctx_mean:.4f},{r.wc_mean:.4f},"
                       f"{r.suvr:.4f},{r.centiloid:.2f},{exp_suvr_str},{exp_cl_str},{diff_str}\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Validate Centiloid pipeline against GAAIN PiB data"
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Directory containing GAAIN PiB NIfTI files and VOI masks"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output CSV file for results"
    )

    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"ERROR: Directory not found: {args.data_dir}")
        sys.exit(1)

    validate_pipeline(args.data_dir, args.output)


if __name__ == "__main__":
    main()
