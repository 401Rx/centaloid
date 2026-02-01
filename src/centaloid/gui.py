"""Browser-based GUI for the Centiloid calculator.

Uses Python's built-in http.server + webbrowser so the app works on any
system regardless of Tcl/Tk version.  All UI is rendered as HTML served
from a local HTTP server on a random port.

Screens
-------
1. **Welcome / DICOM Selection** – enter a folder path and pick a tracer.
2. **Processing** – progress bar while the pipeline runs.
3. **Results** – Centiloid value, SUVr, regional table, classification.
4. **Export** – download TXT / HTML reports.
"""

from __future__ import annotations

import io
import json
import logging
import os
import socket
import tempfile
import threading
import webbrowser
from functools import partial
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

from .centiloid import CentiloidResult, list_supported_tracers
from .pipeline import PipelineConfig, PipelineResult, run_pipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State shared between the HTTP handler and the pipeline thread
# ---------------------------------------------------------------------------

_state: dict = {
    "phase": "welcome",           # welcome | processing | results | error
    "progress_msg": "",
    "progress_frac": 0.0,
    "error": "",
    "result": None,               # PipelineResult | None
    "dicom_dir": "",
    "tracer": "",
    "shutdown_flag": False,
}
_state_lock = threading.Lock()

# ---------------------------------------------------------------------------
# HTML Templates
# ---------------------------------------------------------------------------

_BASE_CSS = """
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                 'Helvetica Neue', Arial, sans-serif;
    background: #f1f5f9; color: #1e293b; line-height: 1.6;
}
.navbar {
    background: #2563eb; color: white; padding: 12px 24px;
    display: flex; align-items: center; gap: 12px;
}
.navbar h1 { font-size: 1.25rem; font-weight: 700; }
.navbar .sub { font-size: 0.85rem; color: #bfdbfe; }
.container { max-width: 820px; margin: 2rem auto; padding: 0 1rem; }
.card {
    background: #fff; border-radius: 8px; padding: 1.5rem;
    margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}
.card h2 { font-size: 1.1rem; margin-bottom: 0.75rem; color: #334155; }
label { display: block; font-weight: 600; margin-bottom: 0.35rem; color: #475569; font-size: 0.9rem; }
input[type=text], select {
    width: 100%; padding: 0.5rem 0.75rem; border: 1px solid #cbd5e1;
    border-radius: 6px; font-size: 0.95rem; margin-bottom: 1rem;
}
input[type=text]:focus, select:focus {
    outline: none; border-color: #2563eb; box-shadow: 0 0 0 3px rgba(37,99,235,0.15);
}
.btn {
    display: inline-block; padding: 0.6rem 1.5rem; border: none; border-radius: 6px;
    font-size: 0.95rem; font-weight: 600; cursor: pointer; text-decoration: none;
}
.btn-primary { background: #2563eb; color: white; }
.btn-primary:hover { background: #1d4ed8; }
.btn-outline { background: white; color: #2563eb; border: 1px solid #2563eb; }
.btn-outline:hover { background: #eff6ff; }
.progress-container {
    background: #e2e8f0; border-radius: 9999px; height: 24px;
    overflow: hidden; margin: 1rem 0;
}
.progress-fill {
    height: 100%; border-radius: 9999px; transition: width 0.3s ease;
    background: linear-gradient(90deg, #2563eb, #3b82f6);
}
.cl-value { font-size: 3.5rem; font-weight: 700; text-align: center; }
.cl-label { text-align: center; font-size: 0.85rem; color: #64748b; }
.badge {
    display: inline-block; padding: 0.2rem 0.75rem; border-radius: 9999px;
    font-size: 0.85rem; font-weight: 600;
}
.badge-green { background: #dcfce7; color: #166534; }
.badge-yellow { background: #fef9c3; color: #854d0e; }
.badge-orange { background: #ffedd5; color: #9a3412; }
.badge-red { background: #fee2e2; color: #991b1b; }
.stat-row { display: flex; justify-content: center; gap: 3rem; margin: 1rem 0; }
.stat-item { text-align: center; }
.stat-val { font-size: 1.3rem; font-weight: 600; }
.stat-lbl { font-size: 0.8rem; color: #64748b; }
table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
th { text-align: left; padding: 0.5rem; border-bottom: 2px solid #e2e8f0; color: #475569; font-weight: 600; }
td { padding: 0.5rem; border-bottom: 1px solid #f1f5f9; }
tr:hover { background: #f8fafc; }
.warning {
    background: #fffbeb; border-left: 4px solid #f59e0b; padding: 0.75rem 1rem;
    margin-bottom: 1rem; border-radius: 0 4px 4px 0; font-size: 0.9rem; color: #92400e;
}
.bar-container {
    background: #e2e8f0; border-radius: 4px; height: 20px;
    margin: 1rem 0; position: relative;
}
.bar-fill {
    height: 100%; border-radius: 4px;
    background: linear-gradient(90deg, #22c55e 0%, #eab308 33%, #f97316 55%, #ef4444 80%);
    width: 100%;
}
.bar-marker {
    position: absolute; top: -4px; width: 4px; height: 28px;
    background: #0f172a; border-radius: 2px;
}
.bar-labels { display: flex; justify-content: space-between; font-size: 0.75rem; color: #94a3b8; }
.export-row { display: flex; gap: 0.75rem; flex-wrap: wrap; }
.footer { text-align: center; font-size: 0.8rem; color: #94a3b8; margin-top: 1.5rem; }
.center { text-align: center; }
"""

_NAVBAR = """
<div class="navbar">
    <h1>Centaloid</h1>
    <span class="sub">PET-CT Centiloid Calculator</span>
</div>
"""


def _page(title: str, body: str, *, auto_refresh: float = 0) -> str:
    """Wrap body HTML in a full page."""
    refresh_tag = ""
    if auto_refresh > 0:
        refresh_tag = f'<meta http-equiv="refresh" content="{auto_refresh}">'
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
{refresh_tag}
<style>{_BASE_CSS}</style>
</head>
<body>
{_NAVBAR}
{body}
</body>
</html>"""


def _welcome_page() -> str:
    tracer_options = '<option value="">(auto-detect from DICOM)</option>\n'
    for t in list_supported_tracers():
        tracer_options += f'<option value="{t}">{t}</option>\n'

    body = f"""
<div class="container">
    <div class="card center">
        <h2 style="font-size:1.4rem;">Welcome to Centaloid</h2>
        <p style="color:#64748b;margin-bottom:1.5rem;">
            Compute the Centiloid value for a beta-amyloid PET scan
        </p>
    </div>
    <div class="card">
        <h2>Load DICOM Series</h2>
        <form method="POST" action="/start">
            <label for="dicom_dir">Path to DICOM folder:</label>
            <input type="text" id="dicom_dir" name="dicom_dir"
                   placeholder="/path/to/dicom/folder" required>

            <label for="tracer">Tracer:</label>
            <select id="tracer" name="tracer">
                {tracer_options}
            </select>

            <div class="center">
                <button type="submit" class="btn btn-primary">
                    Compute Centiloid &rarr;
                </button>
            </div>
        </form>
    </div>
    <div class="footer">
        <p>Centiloid scale: 0 = young-control mean, 100 = typical-AD mean</p>
        <p>Reference: Klunk WE et al. <em>Alzheimers Dement</em> 2015;11(1):1-15</p>
    </div>
</div>
"""
    return _page("Centaloid – Welcome", body)


def _processing_page() -> str:
    with _state_lock:
        msg = _state["progress_msg"] or "Initialising..."
        frac = _state["progress_frac"]
    pct = max(0, min(100, frac * 100))
    body = f"""
<div class="container">
    <div class="card center">
        <h2 style="font-size:1.4rem;">Computing Centiloid</h2>
        <p style="color:#64748b;margin-bottom:0.5rem;">{msg}</p>
        <div class="progress-container">
            <div class="progress-fill" style="width:{pct:.1f}%"></div>
        </div>
        <p style="color:#64748b;font-size:0.9rem;">{pct:.0f}%</p>
        <p style="color:#94a3b8;font-size:0.8rem;margin-top:1rem;">
            This page refreshes automatically.
        </p>
    </div>
</div>
"""
    return _page("Centaloid – Processing", body, auto_refresh=1)


def _cl_color(cl: float) -> str:
    if cl < 12:
        return "#22c55e"
    elif cl < 20:
        return "#eab308"
    elif cl < 50:
        return "#f97316"
    return "#ef4444"


def _cl_badge(classification: str) -> str:
    lo = classification.lower()
    if "negative" in lo:
        return "badge-green"
    elif "indeterminate" in lo:
        return "badge-yellow"
    elif "moderate" in lo:
        return "badge-orange"
    return "badge-red"


def _results_page(result: PipelineResult) -> str:
    res = result.centiloid_result
    color = _cl_color(res.centiloid)
    badge_cls = _cl_badge(res.classification)
    bar_pct = max(0, min(100, res.centiloid / 1.2))

    warning_html = ""
    if res.confidence_note:
        warning_html = f'<div class="warning">{res.confidence_note}</div>'

    # Info table from the DICOM series
    info = result.pet_volume.info
    info_html = ""
    if info:
        info_html = f"""
    <div class="card">
        <h2>Patient / Study Information</h2>
        <table>
            <tr><td style="font-weight:600;width:40%">Patient ID</td><td>{info.patient_id}</td></tr>
            <tr><td style="font-weight:600">Patient Name</td><td>{info.patient_name}</td></tr>
            <tr><td style="font-weight:600">Study Date</td><td>{info.study_date}</td></tr>
            <tr><td style="font-weight:600">Series Description</td><td>{info.series_description}</td></tr>
            <tr><td style="font-weight:600">Modality</td><td>{info.modality}</td></tr>
            <tr><td style="font-weight:600">Manufacturer</td><td>{info.manufacturer}</td></tr>
            <tr><td style="font-weight:600">Tracer (DICOM)</td><td>{info.tracer_name}</td></tr>
            <tr><td style="font-weight:600">Injected Dose</td><td>{info.tracer_dose_bq / 1e6:.1f} MBq</td></tr>
            <tr><td style="font-weight:600">Patient Weight</td><td>{info.patient_weight_kg:.1f} kg</td></tr>
            <tr><td style="font-weight:600">Slices</td><td>{info.num_slices}</td></tr>
            <tr><td style="font-weight:600">Matrix</td><td>{info.rows} &times; {info.columns}</td></tr>
            <tr><td style="font-weight:600">Pixel Spacing</td><td>{info.pixel_spacing_mm[0]:.2f} &times; {info.pixel_spacing_mm[1]:.2f} mm</td></tr>
            <tr><td style="font-weight:600">Slice Thickness</td><td>{info.slice_thickness_mm:.2f} mm</td></tr>
        </table>
    </div>
"""

    rows_html = ""
    for r in res.regions:
        rows_html += (
            f"<tr><td>{r.label}</td><td>{r.mean_uptake:.4f}</td>"
            f"<td>{r.std_uptake:.4f}</td><td>{r.voxel_count}</td>"
            f"<td>{r.volume_cc:.1f}</td></tr>\n"
        )

    body = f"""
<div class="container">
    {warning_html}

    <div class="card center">
        <div class="cl-label">CENTILOID VALUE</div>
        <div class="cl-value" style="color:{color}">{res.centiloid:.1f}</div>
        <span class="badge {badge_cls}">{res.classification}</span>

        <div class="bar-container">
            <div class="bar-fill"></div>
            <div class="bar-marker" style="left:{bar_pct:.1f}%"></div>
        </div>
        <div class="bar-labels">
            <span>0 CL (Young Control)</span>
            <span>50 CL</span>
            <span>100 CL (Typical AD)</span>
        </div>

        <div class="stat-row">
            <div class="stat-item">
                <div class="stat-val">{res.suvr:.4f}</div>
                <div class="stat-lbl">SUVr (CTX / WC)</div>
            </div>
            <div class="stat-item">
                <div class="stat-val">{res.ctx_mean:.4f}</div>
                <div class="stat-lbl">CTX Mean</div>
            </div>
            <div class="stat-item">
                <div class="stat-val">{res.ref_mean:.4f}</div>
                <div class="stat-lbl">WC Ref Mean</div>
            </div>
        </div>
        <p style="color:#64748b;font-size:0.9rem;">Tracer: {res.tracer}</p>
    </div>

    {info_html}

    <div class="card">
        <h2>Regional Uptake Values</h2>
        <table>
            <thead>
                <tr><th>Region</th><th>Mean</th><th>SD</th><th>Voxels</th><th>Vol (mL)</th></tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>

    <div class="card">
        <h2>Export Report</h2>
        <div class="export-row">
            <a class="btn btn-outline" href="/export/text" download="centiloid_report.txt">
                Download Text Report
            </a>
            <a class="btn btn-outline" href="/export/html" download="centiloid_report.html">
                Download HTML Report
            </a>
            <a class="btn btn-primary" href="/">New Analysis</a>
        </div>
    </div>

    <div class="footer">
        <p>Centiloid scale: 0 = young-control mean, 100 = typical-AD mean</p>
        <p>Reference: Klunk WE et al. <em>Alzheimers Dement</em> 2015;11(1):1-15</p>
    </div>
</div>
"""
    return _page("Centaloid – Results", body)


def _error_page(error_msg: str) -> str:
    body = f"""
<div class="container">
    <div class="card center">
        <h2 style="color:#ef4444;">Error</h2>
        <p style="margin:1rem 0;">{error_msg}</p>
        <a class="btn btn-primary" href="/">Back to Start</a>
    </div>
</div>
"""
    return _page("Centaloid – Error", body)


# ---------------------------------------------------------------------------
# HTTP Request Handler
# ---------------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    """Serves the single-page app."""

    def log_message(self, fmt, *args):
        # Silence default request logging
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            with _state_lock:
                phase = _state["phase"]

            if phase == "welcome":
                self._send_html(_welcome_page())
            elif phase == "processing":
                self._send_html(_processing_page())
            elif phase == "results":
                with _state_lock:
                    result = _state["result"]
                self._send_html(_results_page(result))
            elif phase == "error":
                with _state_lock:
                    err = _state["error"]
                self._send_html(_error_page(err))
            else:
                self._send_html(_welcome_page())

        elif path == "/poll":
            # AJAX polling endpoint (returns JSON)
            with _state_lock:
                data = {
                    "phase": _state["phase"],
                    "progress_msg": _state["progress_msg"],
                    "progress_frac": _state["progress_frac"],
                }
            self._send_json(data)

        elif path == "/export/text":
            with _state_lock:
                result = _state.get("result")
            if result and result.text_report:
                self._send_download(result.text_report, "centiloid_report.txt", "text/plain")
            else:
                self._send_html(_error_page("No report available."))

        elif path == "/export/html":
            with _state_lock:
                result = _state.get("result")
            if result and result.html_report:
                self._send_download(result.html_report, "centiloid_report.html", "text/html")
            else:
                self._send_html(_error_page("No report available."))

        elif path == "/shutdown":
            self._send_html(_page("Centaloid", '<div class="container"><div class="card center"><h2>Server stopped.</h2><p>You can close this tab.</p></div></div>'))
            with _state_lock:
                _state["shutdown_flag"] = True

        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/start":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            params = parse_qs(body)
            dicom_dir = params.get("dicom_dir", [""])[0].strip()
            tracer = params.get("tracer", [""])[0].strip()

            if not dicom_dir or not Path(dicom_dir).is_dir():
                self._send_html(_error_page(
                    f"Invalid directory: <code>{dicom_dir}</code><br>"
                    "Please provide a valid path to a folder containing PET DICOM files."
                ))
                return

            # Start pipeline
            with _state_lock:
                _state["phase"] = "processing"
                _state["progress_msg"] = "Starting..."
                _state["progress_frac"] = 0.0
                _state["dicom_dir"] = dicom_dir
                _state["tracer"] = tracer
                _state["error"] = ""
                _state["result"] = None

            t = threading.Thread(target=_run_pipeline_thread, daemon=True)
            t.start()

            # Redirect to main page (will show processing)
            self.send_response(303)
            self.send_header("Location", "/")
            self.end_headers()
        else:
            self.send_error(404)

    def _send_html(self, html: str):
        data = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, obj):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_download(self, content: str, filename: str, content_type: str):
        data = content.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


# ---------------------------------------------------------------------------
# Pipeline thread
# ---------------------------------------------------------------------------

def _run_pipeline_thread():
    """Run the pipeline in a background thread, updating shared state."""
    with _state_lock:
        dicom_dir = _state["dicom_dir"]
        tracer = _state["tracer"]

    def _progress(msg: str, frac: float):
        with _state_lock:
            _state["progress_msg"] = msg
            _state["progress_frac"] = frac

    try:
        cfg = PipelineConfig(
            tracer_name=tracer,
            apply_suv_scaling=True,
        )
        result = run_pipeline(dicom_dir, config=cfg, progress=_progress)
        with _state_lock:
            _state["result"] = result
            _state["phase"] = "results"
    except Exception as exc:
        logger.exception("Pipeline failed")
        with _state_lock:
            _state["error"] = str(exc)
            _state["phase"] = "error"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _find_free_port() -> int:
    """Find an available port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class CentaloidApp:
    """Browser-based GUI application.

    Call ``app.run()`` to start the local HTTP server and open the browser.
    """

    def __init__(self, port: int = 0):
        self.port = port or _find_free_port()
        self.server: Optional[HTTPServer] = None

    def run(self):
        """Start the server and open the browser.  Blocks until shutdown."""
        # Reset state
        with _state_lock:
            _state["phase"] = "welcome"
            _state["progress_msg"] = ""
            _state["progress_frac"] = 0.0
            _state["error"] = ""
            _state["result"] = None
            _state["shutdown_flag"] = False

        self.server = HTTPServer(("127.0.0.1", self.port), _Handler)
        url = f"http://127.0.0.1:{self.port}"

        print(f"Centaloid server running at {url}")
        print("Press Ctrl+C to stop.\n")

        # Open browser
        webbrowser.open(url)

        try:
            # Serve until Ctrl+C or /shutdown
            while True:
                self.server.handle_request()
                with _state_lock:
                    if _state["shutdown_flag"]:
                        break
        except KeyboardInterrupt:
            print("\nShutting down.")
        finally:
            self.server.server_close()
