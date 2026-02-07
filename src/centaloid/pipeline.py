"""End-to-end processing pipeline: DICOM → Centiloid result.

Orchestrates:
1. DICOM loading  →  PETVolume
2. Optional SUV scaling
3. Spatial normalisation to MNI-152
4. Centiloid computation
5. Report generation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .dicom_loader import PETVolume, load_dicom_directory, load_dicom_series
from .registration import resample_to_mni, estimate_affine_to_mni, MNI_AFFINE
from .centiloid import compute_centiloid, CentiloidResult
from .report import render_text_report, render_html_report

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Progress callback protocol
# ---------------------------------------------------------------------------

ProgressCallback = Optional[Callable[[str, float], None]]
"""Signature: callback(stage_description, fraction_complete_0_to_1)"""


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Configurable options for the processing pipeline."""

    tracer_name: str = ""
    """Override tracer name (auto-detected from DICOM if empty)."""

    apply_suv_scaling: bool = True
    """Convert raw voxel values to SUV(bw) before analysis."""

    registration_max_iter: int = 200
    """Maximum iterations for the affine registration optimiser."""

    skip_registration: bool = False
    """If True, assume the input is already in MNI space."""

    output_dir: Optional[str] = None
    """Directory for saving reports. None = don't save."""

    generate_html_report: bool = True
    generate_text_report: bool = True


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Everything produced by a pipeline run."""

    pet_volume: PETVolume
    mni_volume: np.ndarray
    mni_affine: np.ndarray
    centiloid_result: CentiloidResult
    text_report: str = ""
    html_report: str = ""


def run_pipeline(
    dicom_path: str | Path,
    config: Optional[PipelineConfig] = None,
    progress: ProgressCallback = None,
) -> PipelineResult:
    """Run the full DICOM → Centiloid pipeline.

    Parameters
    ----------
    dicom_path
        Path to a directory of DICOM files or a list of file paths.
    config
        Pipeline options.  Uses defaults if ``None``.
    progress
        Optional callback ``(stage: str, fraction: float) → None``.

    Returns
    -------
    PipelineResult
    """
    if config is None:
        config = PipelineConfig()

    def _progress(msg: str, frac: float):
        if progress is not None:
            progress(msg, frac)

    # 1. Load DICOM -----------------------------------------------------------
    _progress("Loading DICOM series…", 0.0)
    dicom_path = Path(dicom_path)
    pet = load_dicom_directory(dicom_path, require_pet=False)
    logger.info("Loaded %d slices (%s)", pet.info.num_slices, pet.shape)
    _progress("DICOM loaded.", 0.15)

    # 2. SUV scaling ----------------------------------------------------------
    if config.apply_suv_scaling and pet.info.suv_factor != 1.0:
        _progress("Applying SUV(bw) scaling…", 0.20)
        pet.voxel_data = pet.voxel_data * pet.info.suv_factor
        logger.info("Applied SUV factor %.6g", pet.info.suv_factor)
    _progress("SUV scaling done.", 0.25)

    # 3. Spatial normalisation ------------------------------------------------
    if config.skip_registration:
        _progress("Skipping registration (assuming MNI space).", 0.50)
        mni_data = pet.voxel_data
        mni_affine = pet.affine
    else:
        _progress("Registering to MNI-152 space…", 0.30)
        transform = estimate_affine_to_mni(
            pet.voxel_data, pet.affine,
            max_iter=config.registration_max_iter,
        )
        _progress("Resampling…", 0.60)
        mni_data, mni_affine = resample_to_mni(
            pet.voxel_data, pet.affine, transform=transform,
        )
    _progress("Registration complete.", 0.70)

    # 4. Determine tracer -----------------------------------------------------
    tracer = config.tracer_name or pet.info.tracer_name
    if not tracer or tracer.lower() == "unknown":
        tracer = "Florbetaben"  # default fallback
        logger.warning("Could not detect tracer – defaulting to %s", tracer)

    # 5. Centiloid computation ------------------------------------------------
    _progress("Computing Centiloid…", 0.75)
    cl_result = compute_centiloid(mni_data, mni_affine, tracer)
    logger.info("SUVr=%.4f  CL=%.1f  [%s]",
                cl_result.suvr, cl_result.centiloid, cl_result.classification)
    _progress("Centiloid computed.", 0.85)

    # 6. Reports --------------------------------------------------------------
    text_report = ""
    html_report = ""
    if config.generate_text_report:
        text_report = render_text_report(cl_result, pet.info)
    if config.generate_html_report:
        html_report = render_html_report(cl_result, pet.info)

    if config.output_dir:
        out = Path(config.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        if text_report:
            (out / "centiloid_report.txt").write_text(text_report)
        if html_report:
            (out / "centiloid_report.html").write_text(html_report)
        logger.info("Reports saved to %s", out)

    _progress("Pipeline complete.", 1.0)

    return PipelineResult(
        pet_volume=pet,
        mni_volume=mni_data,
        mni_affine=mni_affine,
        centiloid_result=cl_result,
        text_report=text_report,
        html_report=html_report,
    )
