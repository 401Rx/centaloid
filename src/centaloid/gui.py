"""Tkinter-based graphical user interface for the Centiloid calculator.

Screens
-------
1. **Welcome / DICOM Selection** – choose a folder of PET DICOM files.
2. **Series Preview** – thumbnail montage + DICOM metadata summary.
3. **Processing** – animated progress bar during pipeline execution.
4. **Results** – Centiloid value, SUVr, regional table, classification badge.
5. **Report Export** – save TXT / HTML reports and view in browser.
"""

from __future__ import annotations

import logging
import os
import threading
import webbrowser
from pathlib import Path
from tkinter import (
    Tk, Toplevel, Frame, Label, Button, Entry, StringVar, DoubleVar,
    IntVar, BooleanVar, Text, Scrollbar, Canvas, PhotoImage,
    filedialog, messagebox, ttk, END, BOTH, LEFT, RIGHT, TOP, BOTTOM,
    X, Y, W, E, N, S, HORIZONTAL, VERTICAL, WORD, DISABLED, NORMAL,
)
from typing import Optional

import numpy as np

from .centiloid import CentiloidResult, list_supported_tracers
from .dicom_loader import PETVolume, DicomSeriesInfo, discover_dicom_files, load_dicom_series
from .pipeline import PipelineConfig, PipelineResult, run_pipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
BG = "#f8fafc"
BG_CARD = "#ffffff"
FG = "#1e293b"
FG_MUTED = "#64748b"
ACCENT = "#2563eb"
ACCENT_HOVER = "#1d4ed8"
GREEN = "#22c55e"
YELLOW = "#eab308"
ORANGE = "#f97316"
RED = "#ef4444"
BORDER = "#e2e8f0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cl_color(cl: float) -> str:
    if cl < 12:
        return GREEN
    elif cl < 20:
        return YELLOW
    elif cl < 50:
        return ORANGE
    return RED


class _Card(Frame):
    """A rounded-looking card container."""
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BG_CARD, highlightbackground=BORDER,
                         highlightthickness=1, padx=16, pady=12, **kw)


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------

class CentaloidApp:
    """Main GUI application."""

    def __init__(self):
        self.root = Tk()
        self.root.title("Centaloid – PET-CT Centiloid Calculator")
        self.root.geometry("900x700")
        self.root.minsize(750, 550)
        self.root.configure(bg=BG)

        # State
        self.dicom_dir: Optional[Path] = None
        self.pet_volume: Optional[PETVolume] = None
        self.pipeline_result: Optional[PipelineResult] = None
        self.progress_var = DoubleVar(value=0.0)
        self.status_var = StringVar(value="Ready")
        self.tracer_var = StringVar(value="(auto-detect)")

        # Build UI
        self._build_navbar()
        self.content = Frame(self.root, bg=BG)
        self.content.pack(fill=BOTH, expand=True, padx=20, pady=(0, 20))
        self._show_welcome_screen()

    # ----- Navigation bar ---------------------------------------------------

    def _build_navbar(self):
        nav = Frame(self.root, bg=ACCENT, height=50)
        nav.pack(fill=X)
        nav.pack_propagate(False)
        Label(nav, text="Centaloid", font=("Helvetica", 16, "bold"),
              bg=ACCENT, fg="white").pack(side=LEFT, padx=16)
        Label(nav, text="PET-CT Centiloid Calculator", font=("Helvetica", 10),
              bg=ACCENT, fg="#bfdbfe").pack(side=LEFT)
        # Status
        Label(nav, textvariable=self.status_var, font=("Helvetica", 9),
              bg=ACCENT, fg="#93c5fd").pack(side=RIGHT, padx=16)

    # ----- Screen 1: Welcome / DICOM selection ------------------------------

    def _clear_content(self):
        for w in self.content.winfo_children():
            w.destroy()

    def _show_welcome_screen(self):
        self._clear_content()
        self.status_var.set("Select DICOM folder")

        # Centre card
        card = _Card(self.content)
        card.place(relx=0.5, rely=0.45, anchor="center")

        Label(card, text="Welcome to Centaloid", font=("Helvetica", 20, "bold"),
              bg=BG_CARD, fg=FG).pack(pady=(8, 4))
        Label(card, text="Compute the Centiloid value for a beta-amyloid PET scan",
              font=("Helvetica", 11), bg=BG_CARD, fg=FG_MUTED).pack()

        sep = Frame(card, bg=BORDER, height=1)
        sep.pack(fill=X, pady=16)

        Label(card, text="Step 1: Select a folder containing PET DICOM files",
              font=("Helvetica", 10), bg=BG_CARD, fg=FG).pack(anchor=W, pady=(0, 8))

        row = Frame(card, bg=BG_CARD)
        row.pack(fill=X, pady=(0, 12))
        self._dir_entry_var = StringVar()
        Entry(row, textvariable=self._dir_entry_var, width=50,
              font=("Helvetica", 10)).pack(side=LEFT, fill=X, expand=True, padx=(0, 8))
        Button(row, text="Browse…", command=self._browse_dicom,
               bg=ACCENT, fg="white", font=("Helvetica", 10, "bold"),
               activebackground=ACCENT_HOVER, activeforeground="white",
               relief="flat", padx=16, pady=4).pack(side=LEFT)

        # Tracer override
        Label(card, text="Tracer (leave blank to auto-detect from DICOM):",
              font=("Helvetica", 10), bg=BG_CARD, fg=FG).pack(anchor=W, pady=(8, 4))
        tracer_row = Frame(card, bg=BG_CARD)
        tracer_row.pack(fill=X, pady=(0, 12))
        combo = ttk.Combobox(tracer_row, textvariable=self.tracer_var, width=30,
                             values=["(auto-detect)"] + list_supported_tracers(),
                             state="readonly")
        combo.pack(side=LEFT)

        Button(card, text="Load DICOM Series  →", command=self._load_dicom,
               bg=ACCENT, fg="white", font=("Helvetica", 11, "bold"),
               activebackground=ACCENT_HOVER, activeforeground="white",
               relief="flat", padx=24, pady=6).pack(pady=(8, 4))

    def _browse_dicom(self):
        d = filedialog.askdirectory(title="Select PET DICOM Folder")
        if d:
            self._dir_entry_var.set(d)

    def _load_dicom(self):
        path = self._dir_entry_var.get().strip()
        if not path or not Path(path).is_dir():
            messagebox.showerror("Error", "Please select a valid directory.")
            return
        self.dicom_dir = Path(path)
        try:
            files = discover_dicom_files(self.dicom_dir)
            if not files:
                messagebox.showerror("Error", "No DICOM files found in the selected folder.")
                return
            self.pet_volume = load_dicom_series(files, require_pet=False)
        except Exception as exc:
            messagebox.showerror("DICOM Error", str(exc))
            return
        self._show_preview_screen()

    # ----- Screen 2: Series Preview -----------------------------------------

    def _show_preview_screen(self):
        self._clear_content()
        self.status_var.set("Review series")
        vol = self.pet_volume
        info = vol.info

        # Header
        hdr = Frame(self.content, bg=BG)
        hdr.pack(fill=X, pady=(0, 8))
        Label(hdr, text="Series Preview", font=("Helvetica", 16, "bold"),
              bg=BG, fg=FG).pack(side=LEFT)
        Button(hdr, text="← Back", command=self._show_welcome_screen,
               bg=BG_CARD, fg=ACCENT, relief="flat",
               font=("Helvetica", 10)).pack(side=RIGHT)

        # Two-column layout
        cols = Frame(self.content, bg=BG)
        cols.pack(fill=BOTH, expand=True)

        # Left: montage
        left = _Card(cols)
        left.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 8))
        Label(left, text="Slice Montage (axial)", font=("Helvetica", 11, "bold"),
              bg=BG_CARD, fg=FG).pack(anchor=W, pady=(0, 8))
        canvas = Canvas(left, bg="#0f172a", highlightthickness=0)
        canvas.pack(fill=BOTH, expand=True)
        self.root.update_idletasks()
        self._draw_montage(canvas, vol.voxel_data)

        # Right: metadata table
        right = _Card(cols)
        right.pack(side=RIGHT, fill=Y, padx=(8, 0))
        Label(right, text="DICOM Metadata", font=("Helvetica", 11, "bold"),
              bg=BG_CARD, fg=FG).pack(anchor=W, pady=(0, 8))

        fields = [
            ("Patient ID", info.patient_id),
            ("Patient Name", info.patient_name),
            ("Study Date", info.study_date),
            ("Modality", info.modality),
            ("Tracer", info.tracer_name),
            ("Manufacturer", info.manufacturer),
            ("Slices", str(info.num_slices)),
            ("Matrix", f"{info.rows}×{info.columns}"),
            ("Pixel Spacing", f"{info.pixel_spacing_mm[0]:.2f}×{info.pixel_spacing_mm[1]:.2f} mm"),
            ("Slice Thickness", f"{info.slice_thickness_mm:.2f} mm"),
            ("Dose", f"{info.tracer_dose_bq/1e6:.1f} MBq"),
            ("Weight", f"{info.patient_weight_kg:.1f} kg"),
            ("SUV Factor", f"{info.suv_factor:.6g}"),
        ]
        for label, value in fields:
            row = Frame(right, bg=BG_CARD)
            row.pack(fill=X, pady=1)
            Label(row, text=label, font=("Helvetica", 9, "bold"),
                  bg=BG_CARD, fg=FG_MUTED, width=14, anchor=W).pack(side=LEFT)
            Label(row, text=value or "—", font=("Helvetica", 9),
                  bg=BG_CARD, fg=FG, anchor=W).pack(side=LEFT, fill=X)

        # Action buttons
        btns = Frame(self.content, bg=BG)
        btns.pack(fill=X, pady=(12, 0))
        Button(btns, text="Compute Centiloid  →", command=self._start_processing,
               bg=ACCENT, fg="white", font=("Helvetica", 12, "bold"),
               activebackground=ACCENT_HOVER, activeforeground="white",
               relief="flat", padx=24, pady=8).pack(side=RIGHT)

    def _draw_montage(self, canvas: Canvas, data: np.ndarray):
        """Draw a simple montage of axial slices onto *canvas*."""
        canvas.update_idletasks()
        cw, ch = canvas.winfo_width(), canvas.winfo_height()
        if cw < 10 or ch < 10:
            cw, ch = 400, 300

        nz = data.shape[0]
        # Pick ~16 evenly spaced slices
        n_show = min(16, nz)
        indices = np.linspace(0, nz - 1, n_show, dtype=int)
        cols_n = 4
        rows_n = int(np.ceil(n_show / cols_n))

        tile_w = cw // cols_n
        tile_h = ch // rows_n

        # Normalise
        dmin, dmax = float(data.min()), float(data.max())
        if dmax - dmin < 1e-12:
            dmax = dmin + 1

        self._montage_images = []  # prevent GC
        for idx, sl in enumerate(indices):
            r, c = divmod(idx, cols_n)
            slc = data[sl]
            # Downsample
            step_r = max(1, slc.shape[0] // tile_h)
            step_c = max(1, slc.shape[1] // tile_w)
            thumb = slc[::step_r, ::step_c]
            # Scale to 0-255
            thumb_u8 = ((thumb - dmin) / (dmax - dmin) * 255).clip(0, 255).astype(np.uint8)
            h, w = thumb_u8.shape
            # PPM header
            header = f"P5 {w} {h} 255 ".encode()
            ppm = header + thumb_u8.tobytes()
            try:
                img = PhotoImage(data=ppm, master=canvas)
            except Exception:
                # Fallback: create a small placeholder
                img = PhotoImage(width=tile_w, height=tile_h, master=canvas)
            self._montage_images.append(img)
            x = c * tile_w + tile_w // 2
            y = r * tile_h + tile_h // 2
            canvas.create_image(x, y, image=img)

    # ----- Screen 3: Processing ---------------------------------------------

    def _start_processing(self):
        self._clear_content()
        self.status_var.set("Processing…")

        card = _Card(self.content)
        card.place(relx=0.5, rely=0.45, anchor="center")

        Label(card, text="Computing Centiloid", font=("Helvetica", 18, "bold"),
              bg=BG_CARD, fg=FG).pack(pady=(8, 4))
        self._proc_label = Label(card, text="Initialising…",
                                 font=("Helvetica", 10), bg=BG_CARD, fg=FG_MUTED)
        self._proc_label.pack(pady=(0, 12))

        self._prog_bar = ttk.Progressbar(card, length=400, mode="determinate",
                                          variable=self.progress_var)
        self._prog_bar.pack(pady=(0, 8))
        self._pct_label = Label(card, text="0 %", font=("Helvetica", 10),
                                bg=BG_CARD, fg=FG_MUTED)
        self._pct_label.pack()

        # Run pipeline in background thread
        self.progress_var.set(0)
        t = threading.Thread(target=self._run_pipeline_thread, daemon=True)
        t.start()

    def _pipeline_progress(self, msg: str, frac: float):
        """Called from the pipeline thread."""
        self.root.after(0, self._update_progress, msg, frac)

    def _update_progress(self, msg: str, frac: float):
        self.progress_var.set(frac * 100)
        self._proc_label.config(text=msg)
        self._pct_label.config(text=f"{frac*100:.0f} %")

    def _run_pipeline_thread(self):
        try:
            tracer = self.tracer_var.get()
            if tracer == "(auto-detect)":
                tracer = ""

            cfg = PipelineConfig(
                tracer_name=tracer,
                apply_suv_scaling=True,
            )
            result = run_pipeline(
                self.dicom_dir,
                config=cfg,
                progress=self._pipeline_progress,
            )
            self.pipeline_result = result
            self.root.after(0, self._show_results_screen)
        except Exception as exc:
            self.root.after(0, lambda: messagebox.showerror("Pipeline Error", str(exc)))
            self.root.after(0, self._show_welcome_screen)

    # ----- Screen 4: Results ------------------------------------------------

    def _show_results_screen(self):
        self._clear_content()
        self.status_var.set("Results ready")
        res = self.pipeline_result.centiloid_result

        # Scrollable frame
        canvas = Canvas(self.content, bg=BG, highlightthickness=0)
        scrollbar = Scrollbar(self.content, orient=VERTICAL, command=canvas.yview)
        scroll_frame = Frame(canvas, bg=BG)
        scroll_frame.bind("<Configure>",
                          lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y)

        # Header
        hdr = Frame(scroll_frame, bg=BG)
        hdr.pack(fill=X, pady=(0, 8))
        Label(hdr, text="Centiloid Analysis Results", font=("Helvetica", 16, "bold"),
              bg=BG, fg=FG).pack(side=LEFT)
        Button(hdr, text="New Analysis", command=self._show_welcome_screen,
               bg=BG_CARD, fg=ACCENT, relief="flat",
               font=("Helvetica", 10)).pack(side=RIGHT)

        # Big result card
        rcard = _Card(scroll_frame)
        rcard.pack(fill=X, pady=(0, 12))

        cl_color = _cl_color(res.centiloid)
        Label(rcard, text="CENTILOID VALUE", font=("Helvetica", 9),
              bg=BG_CARD, fg=FG_MUTED).pack()
        Label(rcard, text=f"{res.centiloid:.1f}",
              font=("Helvetica", 42, "bold"), bg=BG_CARD, fg=cl_color).pack()
        Label(rcard, text=res.classification, font=("Helvetica", 12, "bold"),
              bg=BG_CARD, fg=cl_color).pack(pady=(0, 8))

        # SUVr row
        srow = Frame(rcard, bg=BG_CARD)
        srow.pack(pady=(8, 0))
        for label, val in [("SUVr", res.suvr), ("CTX Mean", res.ctx_mean),
                           ("WC Ref", res.ref_mean)]:
            f = Frame(srow, bg=BG_CARD, padx=24)
            f.pack(side=LEFT)
            Label(f, text=f"{val:.4f}", font=("Helvetica", 16, "bold"),
                  bg=BG_CARD, fg=FG).pack()
            Label(f, text=label, font=("Helvetica", 9), bg=BG_CARD, fg=FG_MUTED).pack()

        # Tracer
        Label(rcard, text=f"Tracer: {res.tracer}", font=("Helvetica", 10),
              bg=BG_CARD, fg=FG_MUTED).pack(pady=(12, 0))

        if res.confidence_note:
            wcard = Frame(scroll_frame, bg="#fffbeb", padx=12, pady=8,
                          highlightbackground="#f59e0b", highlightthickness=2)
            wcard.pack(fill=X, pady=(0, 12))
            Label(wcard, text=res.confidence_note, font=("Helvetica", 9),
                  bg="#fffbeb", fg="#92400e", wraplength=700, justify=LEFT).pack()

        # Regional table
        tcard = _Card(scroll_frame)
        tcard.pack(fill=X, pady=(0, 12))
        Label(tcard, text="Regional Uptake Values", font=("Helvetica", 12, "bold"),
              bg=BG_CARD, fg=FG).pack(anchor=W, pady=(0, 8))

        # Table header
        hdr_frame = Frame(tcard, bg=BORDER)
        hdr_frame.pack(fill=X)
        for col, w in [("Region", 250), ("Mean", 80), ("SD", 80),
                        ("Voxels", 80), ("Vol (mL)", 80)]:
            Label(hdr_frame, text=col, font=("Helvetica", 9, "bold"),
                  bg=BORDER, fg=FG, width=w // 8, anchor=W).pack(side=LEFT, padx=4, pady=4)

        for r in res.regions:
            rf = Frame(tcard, bg=BG_CARD)
            rf.pack(fill=X)
            for text, w in [(r.label, 250), (f"{r.mean_uptake:.4f}", 80),
                            (f"{r.std_uptake:.4f}", 80), (str(r.voxel_count), 80),
                            (f"{r.volume_cc:.1f}", 80)]:
                Label(rf, text=text, font=("Helvetica", 9), bg=BG_CARD, fg=FG,
                      width=w // 8, anchor=W).pack(side=LEFT, padx=4, pady=2)
            Frame(tcard, bg=BORDER, height=1).pack(fill=X)

        # Export buttons
        ecard = _Card(scroll_frame)
        ecard.pack(fill=X)
        Label(ecard, text="Export Report", font=("Helvetica", 12, "bold"),
              bg=BG_CARD, fg=FG).pack(anchor=W, pady=(0, 8))
        btn_row = Frame(ecard, bg=BG_CARD)
        btn_row.pack(fill=X)

        Button(btn_row, text="Save Text Report", command=self._save_text_report,
               bg=BG_CARD, fg=ACCENT, relief="groove",
               font=("Helvetica", 10), padx=12, pady=4).pack(side=LEFT, padx=(0, 8))
        Button(btn_row, text="Save HTML Report", command=self._save_html_report,
               bg=BG_CARD, fg=ACCENT, relief="groove",
               font=("Helvetica", 10), padx=12, pady=4).pack(side=LEFT, padx=(0, 8))
        Button(btn_row, text="View in Browser", command=self._view_html_browser,
               bg=ACCENT, fg="white", relief="flat",
               font=("Helvetica", 10, "bold"), padx=12, pady=4).pack(side=LEFT)

    # ----- Export actions ---------------------------------------------------

    def _save_text_report(self):
        if not self.pipeline_result:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")],
            initialfile="centiloid_report.txt",
        )
        if path:
            Path(path).write_text(self.pipeline_result.text_report)
            messagebox.showinfo("Saved", f"Text report saved to:\n{path}")

    def _save_html_report(self):
        if not self.pipeline_result:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".html",
            filetypes=[("HTML files", "*.html")],
            initialfile="centiloid_report.html",
        )
        if path:
            Path(path).write_text(self.pipeline_result.html_report)
            messagebox.showinfo("Saved", f"HTML report saved to:\n{path}")

    def _view_html_browser(self):
        if not self.pipeline_result:
            return
        import tempfile
        tmp = Path(tempfile.mktemp(suffix=".html"))
        tmp.write_text(self.pipeline_result.html_report)
        webbrowser.open(tmp.as_uri())

    # ----- Run --------------------------------------------------------------

    def run(self):
        """Start the Tk main loop."""
        self.root.mainloop()
