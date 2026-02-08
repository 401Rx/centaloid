"""Entry points for the Centaloid application (CLI and GUI)."""

from __future__ import annotations

import os
import argparse
import logging
import sys
from pathlib import Path

# Force matplotlib to use a non-interactive backend before anything imports
# it.  On some macOS versions the default "macosx" backend crashes with
# "macOS XX or later required".  Agg is a safe headless rasterizer.
os.environ.setdefault("MPLBACKEND", "Agg")
try:
    import matplotlib
    matplotlib.use("Agg", force=True)
except ImportError:
    pass

from . import __version__


def _setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        prog="centaloid",
        description="Centaloid – PET-CT Centiloid Calculator for Beta-Amyloid Imaging",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    sub = parser.add_subparsers(dest="command")

    # --- 'run' sub-command ---
    run_p = sub.add_parser("run", help="Run the Centiloid pipeline on a DICOM folder")
    run_p.add_argument("dicom_dir", type=str, help="Path to folder of PET DICOM files")
    run_p.add_argument("--tracer", type=str, default="",
                       help="Tracer name (auto-detected if omitted)")
    run_p.add_argument("--output", "-o", type=str, default=None,
                       help="Output directory for reports")
    run_p.add_argument("--no-suv", action="store_true",
                       help="Skip SUV(bw) scaling")
    run_p.add_argument("--skip-registration", action="store_true",
                       help="Assume data is already in MNI space")
    run_p.add_argument("--dicom-sc", action="store_true",
                       help="Generate DICOM Secondary Capture image")
    run_p.add_argument("--dicom-sr", action="store_true",
                       help="Generate DICOM Structured Report")
    run_p.add_argument("--verbose", "-v", action="store_true")

    # --- 'tracers' sub-command ---
    sub.add_parser("tracers", help="List supported tracers")

    # --- 'gui' sub-command ---
    sub.add_parser("gui", help="Launch the graphical interface")

    args = parser.parse_args()

    if args.command is None:
        # Default: launch GUI
        main_gui()
        return

    if args.command == "gui":
        main_gui()
        return

    if args.command == "tracers":
        from .centiloid import list_supported_tracers
        print("Supported amyloid PET tracers:")
        for t in list_supported_tracers():
            print(f"  • {t}")
        return

    if args.command == "run":
        _setup_logging(args.verbose)
        from .pipeline import PipelineConfig, run_pipeline

        cfg = PipelineConfig(
            tracer_name=args.tracer,
            apply_suv_scaling=not args.no_suv,
            skip_registration=args.skip_registration,
            output_dir=args.output,
            generate_dicom_sc=args.dicom_sc,
            generate_dicom_sr=args.dicom_sr,
        )

        def _progress(msg: str, frac: float):
            bar_len = 30
            filled = int(bar_len * frac)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"\r  [{bar}] {frac*100:5.1f}%  {msg:<40}", end="", flush=True)

        print(f"Centaloid v{__version__}")
        print(f"DICOM folder: {args.dicom_dir}")
        print()

        try:
            result = run_pipeline(args.dicom_dir, config=cfg, progress=_progress)
        except Exception as exc:
            print(f"\nError: {exc}", file=sys.stderr)
            sys.exit(1)

        print("\n")
        print(result.text_report)

        if args.output:
            print(f"\nReports saved to: {args.output}")

        if result.dicom_files:
            print(f"\nDICOM files created:")
            for f in result.dicom_files:
                print(f"  {f}")


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

def main_gui():
    """Launch the browser-based GUI."""
    _setup_logging(verbose=False)
    from .gui import CentaloidApp
    app = CentaloidApp()
    app.run()


if __name__ == "__main__":
    main()
