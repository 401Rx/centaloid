"""Tests for DICOM loader (using synthetic data)."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from centaloid.dicom_loader import (
    DicomSeriesInfo,
    PETVolume,
    discover_dicom_files,
    load_dicom_series,
)


def _create_synthetic_dicom(tmp_dir: Path, num_slices: int = 10):
    """Create minimal synthetic DICOM files for testing."""
    try:
        import pydicom
        from pydicom.dataset import Dataset, FileDataset
        from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    except ImportError:
        pytest.skip("pydicom not installed")

    files = []
    for i in range(num_slices):
        fname = tmp_dir / f"slice_{i:04d}.dcm"
        ds = FileDataset(str(fname), {}, preamble=b"\x00" * 128, is_implicit_VR=False)
        ds.file_meta = pydicom.Dataset()
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.128"
        ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()

        ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.128"
        ds.SOPInstanceUID = generate_uid()
        ds.Modality = "PT"
        ds.PatientID = "TEST001"
        ds.PatientName = "Test^Patient"
        ds.StudyDate = "20240101"
        ds.Rows = 64
        ds.Columns = 64
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.SamplesPerPixel = 1
        ds.PixelRepresentation = 0
        ds.RescaleSlope = 1.0
        ds.RescaleIntercept = 0.0
        ds.ImagePositionPatient = [0.0, 0.0, float(i * 2.0)]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.PixelSpacing = [2.0, 2.0]
        ds.SliceThickness = 2.0
        ds.PhotometricInterpretation = "MONOCHROME2"

        pixel_data = np.random.randint(0, 4000, (64, 64), dtype=np.uint16)
        ds.PixelData = pixel_data.tobytes()

        ds.save_as(str(fname))
        files.append(fname)
    return files


class TestDiscoverDicomFiles:
    def test_finds_dcm_files(self, tmp_path):
        (tmp_path / "a.dcm").touch()
        (tmp_path / "b.dcm").touch()
        (tmp_path / "c.txt").touch()
        found = discover_dicom_files(tmp_path)
        assert len(found) >= 2

    def test_missing_dir_raises(self):
        with pytest.raises(FileNotFoundError):
            discover_dicom_files("/nonexistent/path")


class TestLoadDicomSeries:
    def test_load_synthetic(self, tmp_path):
        files = _create_synthetic_dicom(tmp_path, num_slices=5)
        vol = load_dicom_series(files, require_pet=True)
        assert isinstance(vol, PETVolume)
        assert vol.shape == (5, 64, 64)
        assert vol.info.patient_id == "TEST001"
        assert vol.info.modality == "PT"

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            load_dicom_series([], require_pet=False)


class TestPETVolume:
    def test_voxel_size(self):
        info = DicomSeriesInfo(
            slice_thickness_mm=2.0,
            pixel_spacing_mm=(1.5, 1.5),
        )
        vol = PETVolume(
            voxel_data=np.zeros((10, 64, 64)),
            affine=np.eye(4),
            info=info,
        )
        assert vol.voxel_size_mm == (2.0, 1.5, 1.5)
