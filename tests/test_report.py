"""Tests for report rendering."""

from centaloid.centiloid import CentiloidResult, RegionResult
from centaloid.dicom_loader import DicomSeriesInfo
from centaloid.report import render_text_report, render_html_report


def _make_result() -> CentiloidResult:
    return CentiloidResult(
        tracer="Florbetapir",
        suvr=1.25,
        centiloid=50.1,
        ctx_mean=1500.0,
        ref_mean=1200.0,
        regions=[
            RegionResult("ctx_composite", "Cortical Composite", 1500.0, 200.0, 50000, 400.0),
            RegionResult("cerebellum_wc", "Whole Cerebellum", 1200.0, 150.0, 30000, 240.0),
        ],
    )


class TestTextReport:
    def test_contains_centiloid(self):
        report = render_text_report(_make_result())
        assert "50.1" in report
        assert "Florbetapir" in report
        assert "1.2500" in report

    def test_with_patient_info(self):
        info = DicomSeriesInfo(patient_id="P001", patient_name="Doe^John")
        report = render_text_report(_make_result(), info)
        assert "P001" in report
        assert "Doe^John" in report


class TestHtmlReport:
    def test_is_valid_html(self):
        html = render_html_report(_make_result())
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html

    def test_contains_values(self):
        html = render_html_report(_make_result())
        assert "50.1" in html
        assert "Florbetapir" in html
