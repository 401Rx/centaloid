"""DICOM output: Save Centiloid results as DICOM files for PACS integration.

This module creates DICOM Secondary Capture (SC) images and/or Structured
Reports (SR) from Centiloid analysis results, enabling seamless integration
with OsiriX, PACS systems, and clinical workflows.
"""

from __future__ import annotations

import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import pydicom
    from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
    from pydicom.uid import generate_uid, ExplicitVRLittleEndian
    from pydicom.sequence import Sequence
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from .centiloid import CentiloidResult
from .atlas import REFERENCE_REGION_KEYS
from .dicom_loader import DicomSeriesInfo

logger = logging.getLogger(__name__)


# DICOM UIDs for Centiloid plugin
IMPLEMENTATION_CLASS_UID = "1.2.826.0.1.3680043.8.498.1"
IMPLEMENTATION_VERSION = "CENTALOID_1.0"

# Series description prefix
SERIES_DESC_PREFIX = "Centiloid Analysis"


def _generate_series_instance_uid() -> str:
    """Generate a unique Series Instance UID."""
    return generate_uid()


def _generate_sop_instance_uid() -> str:
    """Generate a unique SOP Instance UID."""
    return generate_uid()


def create_dicom_secondary_capture(
    result: CentiloidResult,
    source_info: Optional[DicomSeriesInfo] = None,
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """Create a DICOM Secondary Capture image with Centiloid results.

    Renders the results as a visual report image and wraps it in DICOM SC format.

    Parameters
    ----------
    result : CentiloidResult
        The Centiloid computation results.
    source_info : DicomSeriesInfo, optional
        Original DICOM series info (for patient/study demographics).
    output_path : Path, optional
        Where to save the DICOM file. If None, generates a temp path.

    Returns
    -------
    Path to the created DICOM file, or None if creation failed.
    """
    if not PYDICOM_AVAILABLE:
        logger.error("pydicom not installed - cannot create DICOM output")
        return None

    if not PIL_AVAILABLE:
        logger.error("Pillow not installed - cannot create DICOM SC image")
        return None

    # Create the report image
    img = _render_result_image(result, source_info)
    if img is None:
        return None

    # Convert to grayscale numpy array
    img_gray = img.convert('L')
    pixel_array = np.array(img_gray)

    # Create DICOM dataset
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.7"  # Secondary Capture
    file_meta.MediaStorageSOPInstanceUID = _generate_sop_instance_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = IMPLEMENTATION_CLASS_UID
    file_meta.ImplementationVersionName = IMPLEMENTATION_VERSION

    # Create the dataset
    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)

    # Patient Module
    if source_info:
        ds.PatientName = source_info.patient_name or "Unknown"
        ds.PatientID = source_info.patient_id or "Unknown"
        ds.PatientBirthDate = ""
        ds.PatientSex = ""
    else:
        ds.PatientName = "Unknown"
        ds.PatientID = "Unknown"
        ds.PatientBirthDate = ""
        ds.PatientSex = ""

    # Study Module
    now = datetime.now()
    if source_info:
        ds.StudyInstanceUID = source_info.study_instance_uid or generate_uid()
        ds.StudyDate = source_info.study_date or now.strftime("%Y%m%d")
        ds.StudyTime = now.strftime("%H%M%S")
        ds.AccessionNumber = ""
        ds.ReferringPhysicianName = ""
        ds.StudyID = source_info.study_description or "1"
        ds.StudyDescription = source_info.study_description or ""
    else:
        ds.StudyInstanceUID = generate_uid()
        ds.StudyDate = now.strftime("%Y%m%d")
        ds.StudyTime = now.strftime("%H%M%S")
        ds.AccessionNumber = ""
        ds.ReferringPhysicianName = ""
        ds.StudyID = "1"
        ds.StudyDescription = ""

    # Series Module
    ds.SeriesInstanceUID = _generate_series_instance_uid()
    ds.SeriesNumber = 9999
    ds.SeriesDescription = f"{SERIES_DESC_PREFIX} - {result.tracer}"
    ds.Modality = "OT"  # Other

    # General Equipment Module
    ds.Manufacturer = "Centaloid"
    ds.ManufacturerModelName = "Centiloid Calculator"
    ds.SoftwareVersions = "1.0"

    # SC Equipment Module
    ds.ConversionType = "WSD"  # Workstation

    # General Image Module
    ds.InstanceNumber = 1
    ds.ContentDate = now.strftime("%Y%m%d")
    ds.ContentTime = now.strftime("%H%M%S")

    # Image Pixel Module
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = pixel_array.shape[0]
    ds.Columns = pixel_array.shape[1]
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = pixel_array.tobytes()

    # SOP Common Module
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SpecificCharacterSet = "ISO_IR 100"

    # Add Centiloid-specific private tags or use Content Sequence
    # Using standard tags where possible
    reference_suvr = ", ".join(
        f"{region.label} SUVr={region.suvr:.4f}"
        for region in result.regions
        if region.name in REFERENCE_REGION_KEYS
    )
    comment_parts = [
        f"Centiloid={result.centiloid:.1f} CL",
        f"SUVr={result.suvr:.4f}",
        f"Tracer={result.tracer}",
        f"Classification={result.classification}",
    ]
    if reference_suvr:
        comment_parts.append(f"Reference SUVr: {reference_suvr}")
    ds.ImageComments = ", ".join(comment_parts)

    # Save the file
    if output_path is None:
        output_path = Path(f"centiloid_sc_{now.strftime('%Y%m%d_%H%M%S')}.dcm")

    ds.save_as(str(output_path), write_like_original=False)
    logger.info("Created DICOM SC: %s", output_path)

    return output_path


def create_dicom_structured_report(
    result: CentiloidResult,
    source_info: Optional[DicomSeriesInfo] = None,
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """Create a DICOM Structured Report with Centiloid measurements.

    Creates a DICOM SR document containing the quantitative Centiloid
    values in a structured, machine-readable format.

    Parameters
    ----------
    result : CentiloidResult
        The Centiloid computation results.
    source_info : DicomSeriesInfo, optional
        Original DICOM series info.
    output_path : Path, optional
        Where to save the DICOM file.

    Returns
    -------
    Path to the created DICOM file, or None if creation failed.
    """
    if not PYDICOM_AVAILABLE:
        logger.error("pydicom not installed - cannot create DICOM SR")
        return None

    # Create DICOM dataset
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.88.11"  # Basic Text SR
    file_meta.MediaStorageSOPInstanceUID = _generate_sop_instance_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = IMPLEMENTATION_CLASS_UID
    file_meta.ImplementationVersionName = IMPLEMENTATION_VERSION

    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)

    # Patient Module
    if source_info:
        ds.PatientName = source_info.patient_name or "Unknown"
        ds.PatientID = source_info.patient_id or "Unknown"
    else:
        ds.PatientName = "Unknown"
        ds.PatientID = "Unknown"
    ds.PatientBirthDate = ""
    ds.PatientSex = ""

    # Study Module
    now = datetime.now()
    if source_info:
        ds.StudyInstanceUID = source_info.study_instance_uid or generate_uid()
        ds.StudyDate = source_info.study_date or now.strftime("%Y%m%d")
        ds.StudyTime = now.strftime("%H%M%S")
        ds.StudyDescription = source_info.study_description or ""
    else:
        ds.StudyInstanceUID = generate_uid()
        ds.StudyDate = now.strftime("%Y%m%d")
        ds.StudyTime = now.strftime("%H%M%S")
        ds.StudyDescription = ""
    ds.AccessionNumber = ""
    ds.ReferringPhysicianName = ""
    ds.StudyID = "1"

    # Series Module
    ds.SeriesInstanceUID = _generate_series_instance_uid()
    ds.SeriesNumber = 9998
    ds.SeriesDescription = f"{SERIES_DESC_PREFIX} SR - {result.tracer}"
    ds.Modality = "SR"

    # General Equipment
    ds.Manufacturer = "Centaloid"
    ds.ManufacturerModelName = "Centiloid Calculator"
    ds.SoftwareVersions = "1.0"

    # SR Document Series Module
    ds.ReferencedPerformedProcedureStepSequence = Sequence([])

    # SR Document General Module
    ds.InstanceNumber = 1
    ds.ContentDate = now.strftime("%Y%m%d")
    ds.ContentTime = now.strftime("%H%M%S")
    ds.VerificationFlag = "UNVERIFIED"
    ds.CompletionFlag = "COMPLETE"

    # SR Document Content Module
    ds.ValueType = "CONTAINER"
    ds.ContinuityOfContent = "SEPARATE"
    ds.ConceptNameCodeSequence = _create_code_sequence(
        "126000", "DCM", "Imaging Measurement Report"
    )

    # Build content tree
    content_sequence = []

    # Add Centiloid value
    content_sequence.append(_create_num_measurement(
        "Centiloid Value", "CL", result.centiloid,
        code_value="126401", code_scheme="DCM"
    ))

    # Add SUVr
    content_sequence.append(_create_num_measurement(
        "SUVr (CTX/WC)", "ratio", result.suvr,
        code_value="126400", code_scheme="DCM"
    ))

    # Add reference region SUVr values
    for region in result.regions:
        if region.name not in REFERENCE_REGION_KEYS:
            continue
        content_sequence.append(_create_num_measurement(
            f"SUVr ({region.label})", "ratio", region.suvr,
            code_value="126400", code_scheme="DCM"
        ))

    # Add tracer name as text
    content_sequence.append(_create_text_content(
        "Tracer", result.tracer,
        code_value="C-B1031", code_scheme="SRT"
    ))

    # Add classification as text
    content_sequence.append(_create_text_content(
        "Classification", result.classification,
        code_value="126410", code_scheme="DCM"
    ))

    ds.ContentSequence = Sequence(content_sequence)

    # SOP Common
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SpecificCharacterSet = "ISO_IR 100"

    # Save
    if output_path is None:
        output_path = Path(f"centiloid_sr_{now.strftime('%Y%m%d_%H%M%S')}.dcm")

    ds.save_as(str(output_path), write_like_original=False)
    logger.info("Created DICOM SR: %s", output_path)

    return output_path


def _create_code_sequence(code_value: str, code_scheme: str, code_meaning: str) -> Sequence:
    """Create a DICOM code sequence item."""
    item = Dataset()
    item.CodeValue = code_value
    item.CodingSchemeDesignator = code_scheme
    item.CodeMeaning = code_meaning
    return Sequence([item])


def _create_num_measurement(
    meaning: str,
    unit: str,
    value: float,
    code_value: str = "000000",
    code_scheme: str = "DCM",
) -> Dataset:
    """Create a numeric measurement content item."""
    item = Dataset()
    item.RelationshipType = "CONTAINS"
    item.ValueType = "NUM"
    item.ConceptNameCodeSequence = _create_code_sequence(code_value, code_scheme, meaning)

    measured_value = Dataset()
    measured_value.NumericValue = f"{value:.4f}"

    unit_code = Dataset()
    unit_code.CodeValue = unit
    unit_code.CodingSchemeDesignator = "UCUM"
    unit_code.CodeMeaning = unit
    measured_value.MeasurementUnitsCodeSequence = Sequence([unit_code])

    item.MeasuredValueSequence = Sequence([measured_value])
    return item


def _create_text_content(
    meaning: str,
    text: str,
    code_value: str = "000000",
    code_scheme: str = "DCM",
) -> Dataset:
    """Create a text content item."""
    item = Dataset()
    item.RelationshipType = "CONTAINS"
    item.ValueType = "TEXT"
    item.ConceptNameCodeSequence = _create_code_sequence(code_value, code_scheme, meaning)
    item.TextValue = text
    return item


def _render_result_image(
    result: CentiloidResult,
    source_info: Optional[DicomSeriesInfo] = None,
    width: int = 800,
    height: int = 600,
) -> Optional[Image.Image]:
    """Render Centiloid results as a PIL Image."""
    if not PIL_AVAILABLE:
        return None

    # Create white background
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)

    # Try to use a nicer font, fall back to default
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        header_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        body_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except (OSError, IOError):
        title_font = ImageFont.load_default()
        header_font = title_font
        body_font = title_font

    y = 20

    # Title
    draw.text((20, y), "CENTILOID ANALYSIS REPORT", font=title_font, fill='black')
    y += 40

    # Date/time
    draw.text((20, y), f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
              font=body_font, fill='gray')
    y += 30

    # Patient info
    if source_info:
        draw.text((20, y), "PATIENT INFORMATION", font=header_font, fill='darkblue')
        y += 25
        draw.text((40, y), f"Patient ID: {source_info.patient_id}", font=body_font, fill='black')
        y += 20
        draw.text((40, y), f"Patient Name: {source_info.patient_name}", font=body_font, fill='black')
        y += 20
        draw.text((40, y), f"Study Date: {source_info.study_date}", font=body_font, fill='black')
        y += 30

    # Centiloid result
    draw.text((20, y), "CENTILOID RESULT", font=header_font, fill='darkblue')
    y += 25

    # Large Centiloid value
    cl_color = 'green' if result.centiloid < 12 else ('orange' if result.centiloid < 50 else 'red')
    draw.text((40, y), f"Centiloid: {result.centiloid:.1f} CL", font=title_font, fill=cl_color)
    y += 35

    draw.text((40, y), f"SUVr (CTX/WC): {result.suvr:.4f}", font=body_font, fill='black')
    y += 20
    draw.text((40, y), f"Tracer: {result.tracer}", font=body_font, fill='black')
    y += 20
    draw.text((40, y), f"Classification: {result.classification}", font=body_font, fill='black')
    y += 30

    # Regional values
    draw.text((20, y), "REGIONAL VALUES", font=header_font, fill='darkblue')
    y += 25

    for region in result.regions:
        details = (
            f"{region.label}: mean={region.mean_uptake:.4f}, "
            f"voxels={region.voxel_count}, vol={region.volume_cc:.1f}mL"
        )
        if region.name in REFERENCE_REGION_KEYS:
            details += f", SUVr={region.suvr:.4f}"
        draw.text((40, y), details, font=body_font, fill='black')
        y += 18

    y += 20

    # Reference
    draw.text((20, y), "Reference: Klunk WE et al. Alzheimers Dement 2015;11(1):1-15",
              font=body_font, fill='gray')

    return img


def save_results_as_dicom(
    result: CentiloidResult,
    source_info: Optional[DicomSeriesInfo] = None,
    output_dir: Optional[Path] = None,
    create_sc: bool = True,
    create_sr: bool = True,
) -> list[Path]:
    """Save Centiloid results as DICOM files.

    Parameters
    ----------
    result : CentiloidResult
        The Centiloid computation results.
    source_info : DicomSeriesInfo, optional
        Original DICOM series info for demographics.
    output_dir : Path, optional
        Directory to save files. Uses current directory if None.
    create_sc : bool
        Create Secondary Capture image.
    create_sr : bool
        Create Structured Report.

    Returns
    -------
    List of paths to created DICOM files.
    """
    if output_dir is None:
        output_dir = Path(".")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    created_files: list[Path] = []

    if create_sc:
        sc_path = output_dir / f"centiloid_sc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.dcm"
        sc_file = create_dicom_secondary_capture(result, source_info, sc_path)
        if sc_file:
            created_files.append(sc_file)

    if create_sr:
        sr_path = output_dir / f"centiloid_sr_{datetime.now().strftime('%Y%m%d_%H%M%S')}.dcm"
        sr_file = create_dicom_structured_report(result, source_info, sr_path)
        if sr_file:
            created_files.append(sr_file)

    return created_files
