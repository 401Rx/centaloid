#!/usr/bin/env python3
"""Full pipeline validation against GAAIN PiB calibration data.

This script validates the complete centaloid pipeline by:
1. Loading GAAIN PiB NIfTI files (already in MNI space)
2. Running them through our compute_centiloid() function
3. Comparing results against expected values

This validates that our VOI masks and computation match the official Centiloid method.

Usage:
    python validate_full_pipeline.py /path/to/gaain/data

The script will also optionally copy the official GAAIN VOI files to our
data directory if they differ from what we have.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import warnings

import numpy as np

# Add parent directory to path to import centaloid
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import nibabel as nib
except ImportError:
    print("ERROR: nibabel required. Install with: pip install nibabel")
    sys.exit(1)

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from centaloid.centiloid import compute_centiloid, compute_suvr
from centaloid import atlas


def load_nifti(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load NIfTI and return (data, affine)."""
    img = nib.load(path)
    return np.asarray(img.dataobj, dtype=np.float32), img.affine


def find_files(data_dir: Path, pattern: str) -> list[Path]:
    """Find files matching pattern."""
    return sorted(data_dir.glob(pattern))


def compare_voi_masks(our_mask: np.ndarray, gaain_mask: np.ndarray, name: str) -> dict:
    """Compare two binary masks and return statistics."""
    our_binary = our_mask > 0
    gaain_binary = gaain_mask > 0

    our_voxels = our_binary.sum()
    gaain_voxels = gaain_binary.sum()

    intersection = (our_binary & gaain_binary).sum()
    union = (our_binary | gaain_binary).sum()

    dice = 2 * intersection / (our_voxels + gaain_voxels) if (our_voxels + gaain_voxels) > 0 else 0
    jaccard = intersection / union if union > 0 else 0

    return {
        'name': name,
        'our_voxels': int(our_voxels),
        'gaain_voxels': int(gaain_voxels),
        'intersection': int(intersection),
        'union': int(union),
        'dice': dice,
        'jaccard': jaccard,
    }


def validate_voi_files(data_dir: Path):
    """Compare our VOI files against official GAAIN VOIs."""
    print("\n" + "=" * 70)
    print("VOI MASK COMPARISON")
    print("=" * 70)

    # Find GAAIN VOI files
    gaain_ctx = None
    gaain_wc = None

    for pattern in ['voi_ctx_2mm.nii*', 'voi_Ctx_2mm.nii*']:
        files = find_files(data_dir, pattern)
        if files:
            gaain_ctx = files[0]
            break

    for pattern in ['voi_WhlCbl_2mm.nii*', 'voi_whlcbl_2mm.nii*', 'voi_WC_2mm.nii*']:
        files = find_files(data_dir, pattern)
        if files:
            gaain_wc = files[0]
            break

    if gaain_ctx is None or gaain_wc is None:
        print("GAAIN VOI files not found in data directory")
        return

    print(f"GAAIN CTX: {gaain_ctx.name}")
    print(f"GAAIN WC:  {gaain_wc.name}")

    # Load GAAIN masks
    gaain_ctx_data, gaain_ctx_affine = load_nifti(gaain_ctx)
    gaain_wc_data, gaain_wc_affine = load_nifti(gaain_wc)

    print(f"\nGAAIN CTX shape: {gaain_ctx_data.shape}, voxels: {(gaain_ctx_data > 0).sum()}")
    print(f"GAAIN WC shape:  {gaain_wc_data.shape}, voxels: {(gaain_wc_data > 0).sum()}")

    # Generate our masks at the same resolution
    our_ctx = atlas.generate_ctx_mask(gaain_ctx_data.shape, gaain_ctx_affine)
    our_wc = atlas.generate_cerebellum_mask(gaain_wc_data.shape, gaain_wc_affine)

    print(f"\nOur CTX voxels: {our_ctx.sum()}")
    print(f"Our WC voxels:  {our_wc.sum()}")

    # Compare
    ctx_stats = compare_voi_masks(our_ctx, gaain_ctx_data, "CTX")
    wc_stats = compare_voi_masks(our_wc, gaain_wc_data, "WC")

    print(f"\nCTX Comparison:")
    print(f"  Dice coefficient: {ctx_stats['dice']:.4f}")
    print(f"  Jaccard index:    {ctx_stats['jaccard']:.4f}")
    print(f"  Intersection:     {ctx_stats['intersection']} voxels")

    print(f"\nWC Comparison:")
    print(f"  Dice coefficient: {wc_stats['dice']:.4f}")
    print(f"  Jaccard index:    {wc_stats['jaccard']:.4f}")
    print(f"  Intersection:     {wc_stats['intersection']} voxels")

    # Check if we're using official VOIs
    if atlas.is_using_official_voi():
        print("\n✓ Pipeline is using official GAAIN VOI files")
    else:
        print("\n✗ Pipeline is using procedural fallback masks")
        print("  To use official VOIs, copy GAAIN files to:")
        print(f"    {atlas.DATA_DIR}/")

    return ctx_stats, wc_stats


def load_expected_values(data_dir: Path) -> dict:
    """Load expected values from SupplementaryTable1.xlsx."""
    if not HAS_PANDAS:
        return {}

    excel_files = list(data_dir.glob("*upplementary*able*.xlsx")) + \
                  list(data_dir.glob("*upplementary*able*.xls"))

    if not excel_files:
        return {}

    excel_path = excel_files[0]
    print(f"Loading expected values from: {excel_path.name}")

    try:
        df = pd.read_excel(excel_path)
        expected = {}

        # Find columns (handle various naming conventions)
        cols = df.columns.tolist()

        # Try to identify columns by content
        for idx, row in df.iterrows():
            # Look for subject identifiers like AD01, YC101
            for col in cols:
                val = str(row[col]).strip()
                if val.startswith(('AD', 'YC')) and any(c.isdigit() for c in val):
                    subject_id = val.split('_')[0] if '_' in val else val

                    # Look for SUVr and CL values in other columns
                    for other_col in cols:
                        if other_col == col:
                            continue
                        try:
                            other_val = float(row[other_col])
                            # Heuristic: SUVr is typically 0.8-3.0, CL is -50 to 150
                            if col not in expected:
                                expected[subject_id] = {}
                            if 0.5 < other_val < 4.0 and 'suvr' not in expected.get(subject_id, {}):
                                expected[subject_id]['suvr'] = other_val
                            elif -100 < other_val < 200:
                                expected[subject_id]['cl'] = other_val
                        except (ValueError, TypeError):
                            pass
                    break

        return expected
    except Exception as e:
        print(f"  Warning: Could not parse Excel: {e}")
        return {}


def validate_subjects(data_dir: Path, output_path: Path = None):
    """Validate pipeline on all subjects."""
    print("\n" + "=" * 70)
    print("SUBJECT VALIDATION")
    print("=" * 70)

    # Find subject files
    ad_files = find_files(data_dir, "AD*_PiB_*.nii*")
    yc_files = find_files(data_dir, "YC*_PiB_*.nii*")

    print(f"Found {len(ad_files)} AD subjects, {len(yc_files)} YC subjects")

    if not ad_files and not yc_files:
        print("No subject files found!")
        return []

    # Load expected values
    expected = load_expected_values(data_dir)
    if expected:
        print(f"Loaded {len(expected)} expected values")

    results = []

    print("\n" + "-" * 70)
    print(f"{'Subject':<10} {'SUVr':>8} {'CL':>8} {'Exp CL':>8} {'Diff':>8} {'Status'}")
    print("-" * 70)

    for filepath in ad_files + yc_files:
        subject_id = filepath.stem.split('_')[0]
        group = 'AD' if subject_id.startswith('AD') else 'YC'

        try:
            pet_data, pet_affine = load_nifti(filepath)

            # Run through our pipeline
            result = compute_centiloid(pet_data, pet_affine, "PiB")

            suvr = result.suvr
            cl = result.centiloid

            # Compare with expected
            exp_cl = expected.get(subject_id, {}).get('cl')
            exp_suvr = expected.get(subject_id, {}).get('suvr')

            diff = cl - exp_cl if exp_cl is not None else None
            status = ""
            if diff is not None:
                if abs(diff) <= 2.0:
                    status = "✓"
                elif abs(diff) <= 5.0:
                    status = "~"
                else:
                    status = "✗"

            exp_cl_str = f"{exp_cl:.2f}" if exp_cl else "N/A"
            diff_str = f"{diff:+.2f}" if diff is not None else ""

            print(f"{subject_id:<10} {suvr:>8.4f} {cl:>8.2f} {exp_cl_str:>8} {diff_str:>8} {status}")

            results.append({
                'subject': subject_id,
                'group': group,
                'suvr': suvr,
                'centiloid': cl,
                'expected_cl': exp_cl,
                'diff': diff,
            })

        except Exception as e:
            print(f"{subject_id:<10} ERROR: {e}")

    print("-" * 70)

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    ad_results = [r for r in results if r['group'] == 'AD']
    yc_results = [r for r in results if r['group'] == 'YC']

    if yc_results:
        yc_suvr = [r['suvr'] for r in yc_results]
        yc_cl = [r['centiloid'] for r in yc_results]
        print(f"\nYoung Controls (n={len(yc_results)}):")
        print(f"  SUVr: {np.mean(yc_suvr):.4f} ± {np.std(yc_suvr):.4f}")
        print(f"  CL:   {np.mean(yc_cl):.2f} ± {np.std(yc_cl):.2f}")
        print(f"  Expected: SUVr ~1.009, CL ~0")

    if ad_results:
        ad_suvr = [r['suvr'] for r in ad_results]
        ad_cl = [r['centiloid'] for r in ad_results]
        print(f"\nAlzheimer's Disease (n={len(ad_results)}):")
        print(f"  SUVr: {np.mean(ad_suvr):.4f} ± {np.std(ad_suvr):.4f}")
        print(f"  CL:   {np.mean(ad_cl):.2f} ± {np.std(ad_cl):.2f}")
        print(f"  Expected: SUVr ~2.068, CL ~100")

    # Validation metrics
    diffs = [r['diff'] for r in results if r['diff'] is not None]
    if diffs:
        diffs = np.array(diffs)
        print(f"\nValidation Metrics:")
        print(f"  Mean difference:  {np.mean(diffs):+.4f} CL")
        print(f"  SD of difference: {np.std(diffs):.4f} CL")
        print(f"  Max abs diff:     {np.max(np.abs(diffs)):.4f} CL")
        print(f"  Within ±2 CL:     {np.sum(np.abs(diffs) <= 2)}/{len(diffs)}")
        print(f"  Within ±5 CL:     {np.sum(np.abs(diffs) <= 5)}/{len(diffs)}")

    # Save to CSV
    if output_path:
        print(f"\nSaving results to: {output_path}")
        with open(output_path, 'w') as f:
            f.write("Subject,Group,SUVr,Centiloid,Expected_CL,Diff\n")
            for r in results:
                exp = f"{r['expected_cl']:.2f}" if r['expected_cl'] else ""
                diff = f"{r['diff']:.2f}" if r['diff'] else ""
                f.write(f"{r['subject']},{r['group']},{r['suvr']:.4f},{r['centiloid']:.2f},{exp},{diff}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Validate centaloid pipeline against GAAIN data")
    parser.add_argument("data_dir", type=Path, help="Directory with GAAIN PiB data")
    parser.add_argument("--output", "-o", type=Path, help="Output CSV path")
    parser.add_argument("--voi-only", action="store_true", help="Only compare VOI masks")

    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"ERROR: Directory not found: {args.data_dir}")
        sys.exit(1)

    print("=" * 70)
    print("CENTALOID PIPELINE VALIDATION")
    print("=" * 70)
    print(f"Data directory: {args.data_dir}")
    print(f"Using official VOI: {atlas.is_using_official_voi()}")

    # Compare VOI masks
    validate_voi_files(args.data_dir)

    if not args.voi_only:
        # Validate subjects
        validate_subjects(args.data_dir, args.output)


if __name__ == "__main__":
    main()
