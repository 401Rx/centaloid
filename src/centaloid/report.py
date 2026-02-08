"""Report generation: plain-text and HTML summaries of Centiloid results."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from .centiloid import CentiloidResult, RegionResult
from .dicom_loader import DicomSeriesInfo
from .atlas import get_voi_status, is_using_official_voi


# ---------------------------------------------------------------------------
# Plain-text report
# ---------------------------------------------------------------------------

def render_text_report(
    result: CentiloidResult,
    info: Optional[DicomSeriesInfo] = None,
) -> str:
    """Render a human-readable plain-text report."""
    lines: list[str] = []
    w = 60

    lines.append("=" * w)
    lines.append("  CENTILOID ANALYSIS REPORT")
    lines.append("=" * w)
    lines.append(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    if info:
        lines.append("-" * w)
        lines.append("  PATIENT / STUDY INFORMATION")
        lines.append("-" * w)
        lines.append(f"  Patient ID     : {info.patient_id}")
        lines.append(f"  Patient Name   : {info.patient_name}")
        lines.append(f"  Study Date     : {info.study_date}")
        lines.append(f"  Study Desc     : {info.study_description}")
        lines.append(f"  Series Desc    : {info.series_description}")
        lines.append(f"  Modality       : {info.modality}")
        lines.append(f"  Manufacturer   : {info.manufacturer}")
        lines.append(f"  Tracer (DICOM) : {info.tracer_name}")
        lines.append(f"  Injected Dose  : {info.tracer_dose_bq / 1e6:.1f} MBq")
        lines.append(f"  Patient Weight : {info.patient_weight_kg:.1f} kg")
        lines.append(f"  Slices         : {info.num_slices}")
        lines.append(f"  Matrix         : {info.rows} × {info.columns}")
        lines.append(f"  Pixel Spacing  : {info.pixel_spacing_mm[0]:.2f} × {info.pixel_spacing_mm[1]:.2f} mm")
        lines.append(f"  Slice Thickness: {info.slice_thickness_mm:.2f} mm")
        lines.append("")

    lines.append("-" * w)
    lines.append("  CENTILOID RESULT")
    lines.append("-" * w)
    lines.append(f"  Tracer Used    : {result.tracer}")
    lines.append(f"  SUVr (CTX/WC)  : {result.suvr:.4f}")
    lines.append(f"  Centiloid (CL) : {result.centiloid:.1f}")
    lines.append(f"  Classification : {result.classification}")
    lines.append(f"  VOI Masks      : {get_voi_status()}")
    lines.append("")

    if result.confidence_note:
        lines.append(f"  ⚠ {result.confidence_note}")
        lines.append("")

    lines.append("-" * w)
    lines.append("  REGIONAL UPTAKE VALUES")
    lines.append("-" * w)
    lines.append(f"  {'Region':<32} {'SUVr':>6} {'Mean':>8} {'SD':>8} {'Voxels':>8} {'Vol(mL)':>8}")
    lines.append(f"  {'-'*32} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for r in result.regions:
        lines.append(
            f"  {r.label:<32} {r.suvr:>6.3f} {r.mean_uptake:>8.4f} "
            f"{r.std_uptake:>8.4f} {r.voxel_count:>8d} {r.volume_cc:>8.1f}"
        )
    lines.append("")
    lines.append("=" * w)
    lines.append("  Centiloid scale: 0 = young-control mean, 100 = typical-AD mean")
    lines.append("  Reference: Klunk WE et al. Alzheimers Dement 2015;11(1):1-15")
    lines.append("=" * w)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def render_html_report(
    result: CentiloidResult,
    info: Optional[DicomSeriesInfo] = None,
) -> str:
    """Render an HTML report suitable for display in a browser or GUI."""
    cl = result.centiloid
    # Color bar position (clamp 0-120 for display)
    bar_pct = max(0, min(100, cl / 1.2))
    if cl < 12:
        bar_color = "#22c55e"
        badge_class = "negative"
    elif cl < 20:
        bar_color = "#eab308"
        badge_class = "indeterminate"
    elif cl < 50:
        bar_color = "#f97316"
        badge_class = "moderate"
    else:
        bar_color = "#ef4444"
        badge_class = "positive"

    rows_html = ""
    for r in result.regions:
        rows_html += (
            f"<tr><td>{r.label}</td><td>{r.suvr:.3f}</td><td>{r.mean_uptake:.4f}</td>"
            f"<td>{r.std_uptake:.4f}</td><td>{r.voxel_count}</td>"
            f"<td>{r.volume_cc:.1f}</td></tr>\n"
        )

    patient_html = ""
    if info:
        patient_html = f"""
        <div class="section">
            <h2>Patient / Study Information</h2>
            <table class="info-table">
                <tr><td>Patient ID</td><td>{info.patient_id}</td></tr>
                <tr><td>Patient Name</td><td>{info.patient_name}</td></tr>
                <tr><td>Study Date</td><td>{info.study_date}</td></tr>
                <tr><td>Study Description</td><td>{info.study_description}</td></tr>
                <tr><td>Series Description</td><td>{info.series_description}</td></tr>
                <tr><td>Modality</td><td>{info.modality}</td></tr>
                <tr><td>Manufacturer</td><td>{info.manufacturer}</td></tr>
                <tr><td>Tracer (DICOM)</td><td>{info.tracer_name}</td></tr>
                <tr><td>Injected Dose</td><td>{info.tracer_dose_bq / 1e6:.1f} MBq</td></tr>
                <tr><td>Patient Weight</td><td>{info.patient_weight_kg:.1f} kg</td></tr>
                <tr><td>Slices</td><td>{info.num_slices}</td></tr>
                <tr><td>Matrix</td><td>{info.rows} &times; {info.columns}</td></tr>
                <tr><td>Pixel Spacing</td><td>{info.pixel_spacing_mm[0]:.2f} &times; {info.pixel_spacing_mm[1]:.2f} mm</td></tr>
                <tr><td>Slice Thickness</td><td>{info.slice_thickness_mm:.2f} mm</td></tr>
            </table>
        </div>
        """

    warning_html = ""
    if result.confidence_note:
        warning_html = f'<div class="warning">{result.confidence_note}</div>'

    voi_status = get_voi_status()
    if not is_using_official_voi():
        warning_html += (
            '<div class="warning">'
            '<strong>VOI Warning:</strong> Using procedural fallback masks. '
            'For accurate clinical results, download official GAAIN VOI files from: '
            '<a href="https://www.gaain.org/centiloid-project">gaain.org/centiloid-project</a>'
            '</div>'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Centiloid Analysis Report</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: #f8fafc; color: #1e293b; line-height: 1.6; padding: 2rem;
    }}
    .container {{ max-width: 800px; margin: 0 auto; }}
    h1 {{ font-size: 1.5rem; margin-bottom: 0.5rem; color: #0f172a; }}
    h2 {{ font-size: 1.1rem; margin-bottom: 0.75rem; color: #334155; }}
    .timestamp {{ color: #64748b; font-size: 0.85rem; margin-bottom: 1.5rem; }}
    .section {{
        background: #fff; border-radius: 8px; padding: 1.5rem;
        margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }}
    .result-box {{
        text-align: center; padding: 2rem;
    }}
    .cl-value {{ font-size: 3rem; font-weight: 700; color: {bar_color}; }}
    .cl-label {{ font-size: 0.9rem; color: #64748b; margin-top: 0.25rem; }}
    .badge {{
        display: inline-block; padding: 0.25rem 0.75rem; border-radius: 9999px;
        font-size: 0.85rem; font-weight: 600; margin-top: 0.5rem;
    }}
    .badge.negative {{ background: #dcfce7; color: #166534; }}
    .badge.indeterminate {{ background: #fef9c3; color: #854d0e; }}
    .badge.moderate {{ background: #ffedd5; color: #9a3412; }}
    .badge.positive {{ background: #fee2e2; color: #991b1b; }}
    .bar-container {{
        background: #e2e8f0; border-radius: 4px; height: 20px;
        margin: 1rem 0; position: relative; overflow: visible;
    }}
    .bar-fill {{
        height: 100%; border-radius: 4px;
        background: linear-gradient(90deg, #22c55e 0%, #eab308 33%, #f97316 55%, #ef4444 80%);
        width: 100%;
    }}
    .bar-marker {{
        position: absolute; top: -4px; width: 4px; height: 28px;
        background: #0f172a; border-radius: 2px;
        left: {bar_pct:.1f}%;
    }}
    .bar-labels {{
        display: flex; justify-content: space-between;
        font-size: 0.75rem; color: #94a3b8;
    }}
    .suvr-row {{
        display: flex; justify-content: center; gap: 3rem; margin-top: 1rem;
    }}
    .suvr-item {{ text-align: center; }}
    .suvr-val {{ font-size: 1.3rem; font-weight: 600; }}
    .suvr-lbl {{ font-size: 0.8rem; color: #64748b; }}
    .info-table {{ width: 100%; border-collapse: collapse; }}
    .info-table td {{ padding: 0.35rem 0.5rem; border-bottom: 1px solid #f1f5f9; }}
    .info-table td:first-child {{ font-weight: 600; width: 40%; color: #475569; }}
    table.regions {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
    table.regions th {{
        text-align: left; padding: 0.5rem; border-bottom: 2px solid #e2e8f0;
        color: #475569; font-weight: 600;
    }}
    table.regions td {{ padding: 0.5rem; border-bottom: 1px solid #f1f5f9; }}
    table.regions tr:hover {{ background: #f8fafc; }}
    .warning {{
        background: #fffbeb; border-left: 4px solid #f59e0b; padding: 0.75rem 1rem;
        margin-bottom: 1rem; border-radius: 0 4px 4px 0; font-size: 0.9rem;
    }}
    .footer {{
        text-align: center; font-size: 0.8rem; color: #94a3b8; margin-top: 2rem;
    }}
</style>
</head>
<body>
<div class="container">
    <h1>Centiloid Analysis Report</h1>
    <div class="timestamp">Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &middot; Tracer: {result.tracer}</div>

    {warning_html}

    <div class="section result-box">
        <div class="cl-label">CENTILOID VALUE</div>
        <div class="cl-value">{result.centiloid:.1f}</div>
        <span class="badge {badge_class}">{result.classification}</span>
        <div class="bar-container">
            <div class="bar-fill"></div>
            <div class="bar-marker"></div>
        </div>
        <div class="bar-labels">
            <span>0 CL (Young Control)</span>
            <span>50 CL</span>
            <span>100 CL (Typical AD)</span>
        </div>
        <div class="suvr-row">
            <div class="suvr-item">
                <div class="suvr-val">{result.suvr:.4f}</div>
                <div class="suvr-lbl">SUVr (CTX / WC)</div>
            </div>
        </div>
    </div>

    {patient_html}

    <div class="section">
        <h2>Regional Uptake Values</h2>
        <table class="regions">
            <thead>
                <tr><th>Region</th><th>SUVr</th><th>Mean</th><th>SD</th><th>Voxels</th><th>Vol (mL)</th></tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>

    <div class="footer">
        <p>Centiloid scale: 0 = young-control mean, 100 = typical-AD mean</p>
        <p>VOI Masks: {voi_status}</p>
        <p>Reference: Klunk WE et al. <em>Alzheimers Dement</em> 2015;11(1):1-15</p>
    </div>
</div>
</body>
</html>"""
    return html
