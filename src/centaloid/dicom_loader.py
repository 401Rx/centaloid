"""DICOM series loading, validation, and 3-D volume reconstruction."""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

try:
    import pydicom
    from pydicom.dataset import Dataset
    from pydicom.errors import InvalidDicomError
except ImportError:
    pydicom = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class DicomSeriesInfo:
    """Metadata extracted from a loaded DICOM PET series."""

    patient_id: str = ""
    patient_name: str = ""
    study_date: str = ""
    study_description: str = ""
    series_description: str = ""
    modality: str = ""
    manufacturer: str = ""
    tracer_name: str = ""
    tracer_dose_bq: float = 0.0
    patient_weight_kg: float = 0.0
    num_slices: int = 0
    rows: int = 0
    columns: int = 0
    pixel_spacing_mm: tuple[float, float] = (1.0, 1.0)
    slice_thickness_mm: float = 1.0
    image_orientation: tuple[float, ...] = ()
    image_position_first: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rescale_slope: float = 1.0
    rescale_intercept: float = 0.0
    suv_factor: float = 1.0


@dataclass
class PETVolume:
    """A reconstructed 3-D PET volume with associated metadata."""

    voxel_data: np.ndarray  # shape (Z, Y, X) – raw or SUV-scaled
    affine: np.ndarray  # 4×4 voxel-to-world (mm, RAS)
    info: DicomSeriesInfo = field(default_factory=DicomSeriesInfo)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.voxel_data.shape

    @property
    def voxel_size_mm(self) -> tuple[float, float, float]:
        return (
            self.info.slice_thickness_mm,
            self.info.pixel_spacing_mm[0],
            self.info.pixel_spacing_mm[1],
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_dicom_file(path: Path) -> bool:
    """Return True if *path* appears to be a valid DICOM file."""
    if pydicom is None:
        raise ImportError("pydicom is required – install with: pip install pydicom")
    try:
        pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        return True
    except Exception:
        return False


def _safe_getattr(ds: "Dataset", attr: str, default=None):
    """Safely read a DICOM attribute, returning *default* on failure."""
    try:
        return getattr(ds, attr, default)
    except Exception:
        return default


def _extract_tracer_name(ds: "Dataset") -> str:
    """Best-effort extraction of the radiotracer name."""
    # Radiopharmaceutical Information Sequence (0054,0016)
    rpi_seq = _safe_getattr(ds, "RadiopharmaceuticalInformationSequence", None)
    if rpi_seq and len(rpi_seq) > 0:
        item = rpi_seq[0]
        name = _safe_getattr(item, "Radiopharmaceutical", "")
        if name:
            return str(name)
        code_seq = _safe_getattr(item, "RadiopharmaceuticalCodeSequence", None)
        if code_seq and len(code_seq) > 0:
            return str(_safe_getattr(code_seq[0], "CodeMeaning", ""))
    # Fallback: series description heuristics
    desc = str(_safe_getattr(ds, "SeriesDescription", "")).lower()
    for tracer in ("pib", "florbetapir", "amyvid", "florbetaben",
                    "neuraceq", "flutemetamol", "vizamyl", "nav4694",
                    "mk6240", "flortaucipir"):
        if tracer in desc:
            return tracer.capitalize()
    return "Unknown"


def _compute_suv_factor(ds: "Dataset") -> float:
    """Compute the Bq/mL → SUV(bw) conversion factor from DICOM tags.

    SUV(bw) = activity_concentration / (injected_dose / body_weight)
    The voxel values (after rescale slope/intercept) are in Bq/mL.
    """
    weight = float(_safe_getattr(ds, "PatientWeight", 0) or 0)
    if weight <= 0:
        return 1.0  # cannot compute

    rpi_seq = _safe_getattr(ds, "RadiopharmaceuticalInformationSequence", None)
    if not rpi_seq or len(rpi_seq) == 0:
        return 1.0
    item = rpi_seq[0]
    dose = float(_safe_getattr(item, "RadionuclideTotalDose", 0) or 0)
    if dose <= 0:
        return 1.0

    # Decay correction – half-life based
    half_life = float(_safe_getattr(item, "RadionuclideHalfLife", 0) or 0)
    if half_life > 0:
        from datetime import datetime
        start_time = _safe_getattr(item, "RadiopharmaceuticalStartDateTime",
                                   _safe_getattr(item, "RadiopharmaceuticalStartTime", None))
        acq_time = _safe_getattr(ds, "AcquisitionTime",
                                 _safe_getattr(ds, "SeriesTime", None))
        if start_time and acq_time:
            try:
                fmt = "%H%M%S" if len(str(acq_time).split(".")[0]) == 6 else "%H%M%S.%f"
                t_acq = datetime.strptime(str(acq_time).split(".")[0], "%H%M%S")
                t_inj = datetime.strptime(str(start_time).split(".")[0][-6:], "%H%M%S")
                delta_s = (t_acq - t_inj).total_seconds()
                if delta_s > 0:
                    import math
                    decay_factor = math.pow(2, -delta_s / half_life)
                    dose *= decay_factor
            except Exception:
                pass

    if dose <= 0:
        return 1.0

    # SUV(bw) = C [Bq/mL] / (dose [Bq] / weight [g])
    # weight in kg → grams
    return 1.0 / (dose / (weight * 1000.0))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def discover_dicom_files(directory: str | Path) -> list[Path]:
    """Recursively discover DICOM files under *directory*."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")
    candidates: list[Path] = []
    for root, _dirs, files in os.walk(directory):
        for fname in files:
            fpath = Path(root) / fname
            if fpath.suffix.lower() in (".dcm", ".ima", ".img", "") or fpath.suffix == "":
                candidates.append(fpath)
    return sorted(candidates)


def load_dicom_series(
    paths: Sequence[str | Path],
    require_pet: bool = True,
) -> PETVolume:
    """Load a PET DICOM series from a sequence of file paths.

    Parameters
    ----------
    paths : sequence of paths
        Individual DICOM slice files belonging to *one* series.
    require_pet : bool
        If True, raise ``ValueError`` when modality is not PT/PET.

    Returns
    -------
    PETVolume
        Reconstructed 3-D volume with voxel data in raw stored units and
        an affine mapping voxels → world (mm, RAS).
    """
    if pydicom is None:
        raise ImportError("pydicom is required")

    # Read all datasets -------------------------------------------------------
    datasets: list[Dataset] = []
    for p in paths:
        try:
            ds = pydicom.dcmread(str(p), force=True)
            datasets.append(ds)
        except (InvalidDicomError, Exception) as exc:
            logger.warning("Skipping %s: %s", p, exc)

    if not datasets:
        raise ValueError("No valid DICOM files could be loaded.")

    # Validate modality -------------------------------------------------------
    modality = str(_safe_getattr(datasets[0], "Modality", "")).upper()
    if require_pet and modality not in ("PT", "PET"):
        raise ValueError(
            f"Expected PET modality, got '{modality}'. "
            "Pass require_pet=False to override."
        )

    # Sort by ImagePositionPatient[2] (axial slice location) ------------------
    def _sort_key(ds: Dataset):
        pos = _safe_getattr(ds, "ImagePositionPatient", [0, 0, 0])
        return float(pos[2])

    datasets.sort(key=_sort_key)

    # Build volume array ------------------------------------------------------
    ref = datasets[0]
    rows = int(_safe_getattr(ref, "Rows", 0))
    cols = int(_safe_getattr(ref, "Columns", 0))
    n_slices = len(datasets)

    volume = np.zeros((n_slices, rows, cols), dtype=np.float64)
    for i, ds in enumerate(datasets):
        slope = float(_safe_getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(_safe_getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        try:
            arr = ds.pixel_array.astype(np.float64)
        except Exception:
            arr = np.zeros((rows, cols), dtype=np.float64)
        volume[i] = arr * slope + intercept

    # Pixel spacing & slice thickness -----------------------------------------
    ps = _safe_getattr(ref, "PixelSpacing", [1.0, 1.0])
    pixel_spacing = (float(ps[0]), float(ps[1]))
    slice_thickness = float(_safe_getattr(ref, "SliceThickness", 1.0) or 1.0)
    # More robust: compute from actual positions
    if n_slices >= 2:
        pos0 = [float(x) for x in _safe_getattr(datasets[0], "ImagePositionPatient", [0, 0, 0])]
        pos1 = [float(x) for x in _safe_getattr(datasets[1], "ImagePositionPatient", [0, 0, 0])]
        computed_thickness = abs(pos1[2] - pos0[2])
        if computed_thickness > 0:
            slice_thickness = computed_thickness

    # Build affine (DICOM LPS → RAS) -----------------------------------------
    ipp = [float(x) for x in _safe_getattr(ref, "ImagePositionPatient", [0, 0, 0])]
    iop = [float(x) for x in _safe_getattr(ref, "ImageOrientationPatient",
                                             [1, 0, 0, 0, 1, 0])]
    row_cos = np.array(iop[0:3])
    col_cos = np.array(iop[3:6])
    slice_cos = np.cross(row_cos, col_cos)

    affine = np.eye(4)
    affine[0:3, 0] = slice_cos * slice_thickness
    affine[0:3, 1] = row_cos * pixel_spacing[0]
    affine[0:3, 2] = col_cos * pixel_spacing[1]
    affine[0:3, 3] = ipp

    # DICOM is LPS; convert to RAS by negating first two rows
    lps_to_ras = np.diag([-1, -1, 1, 1]).astype(float)
    affine = lps_to_ras @ affine

    # Metadata ----------------------------------------------------------------
    suv_factor = _compute_suv_factor(ref)
    info = DicomSeriesInfo(
        patient_id=str(_safe_getattr(ref, "PatientID", "")),
        patient_name=str(_safe_getattr(ref, "PatientName", "")),
        study_date=str(_safe_getattr(ref, "StudyDate", "")),
        study_description=str(_safe_getattr(ref, "StudyDescription", "")),
        series_description=str(_safe_getattr(ref, "SeriesDescription", "")),
        modality=modality,
        manufacturer=str(_safe_getattr(ref, "Manufacturer", "")),
        tracer_name=_extract_tracer_name(ref),
        tracer_dose_bq=float(_safe_getattr(
            (_safe_getattr(ref, "RadiopharmaceuticalInformationSequence", [None]) or [None])[0],
            "RadionuclideTotalDose", 0) or 0),
        patient_weight_kg=float(_safe_getattr(ref, "PatientWeight", 0) or 0),
        num_slices=n_slices,
        rows=rows,
        columns=cols,
        pixel_spacing_mm=pixel_spacing,
        slice_thickness_mm=slice_thickness,
        image_orientation=tuple(iop),
        image_position_first=(ipp[0], ipp[1], ipp[2]),
        rescale_slope=float(_safe_getattr(ref, "RescaleSlope", 1.0) or 1.0),
        rescale_intercept=float(_safe_getattr(ref, "RescaleIntercept", 0.0) or 0.0),
        suv_factor=suv_factor,
    )

    return PETVolume(voxel_data=volume, affine=affine, info=info)


def load_dicom_directory(directory: str | Path, require_pet: bool = True) -> PETVolume:
    """Convenience wrapper: discover + load all DICOM files in *directory*."""
    paths = discover_dicom_files(directory)
    if not paths:
        raise FileNotFoundError(f"No DICOM files found in {directory}")
    return load_dicom_series(paths, require_pet=require_pet)
