================================================================================
                      GAAIN CENTILOID VOI FILES
================================================================================

This directory is for storing the official GAAIN Centiloid VOI (Volume of
Interest) NIfTI files. These files define the standardized brain regions
used in Centiloid quantification.

Without these files, the software uses approximate procedural masks which
may produce inaccurate Centiloid values.


DOWNLOADING OFFICIAL VOI FILES
------------------------------

1. Visit: https://www.gaain.org/centiloid-project

2. Scroll down to the "Downloads" section

3. Download "Centiloid_Std_VOI.zip"

4. Extract the ZIP file

5. Copy the following NIfTI files to THIS directory:
   - CTX VOI file (cortical target region) - name it: voi_ctx_2mm.nii.gz
   - WC VOI file (whole cerebellum reference) - name it: voi_wc_2mm.nii.gz


EXPECTED FILE NAMES
-------------------

The software looks for files with these names (in order):

CTX (Cortical Target):
  - voi_ctx_2mm.nii / voi_ctx_2mm.nii.gz
  - CTX_VOI.nii / CTX_VOI.nii.gz
  - ctx.nii / ctx.nii.gz
  - Centiloid_Ctx_VOI.nii / Centiloid_Ctx_VOI.nii.gz

WC (Whole Cerebellum):
  - voi_wc_2mm.nii / voi_wc_2mm.nii.gz
  - WC_VOI.nii / WC_VOI.nii.gz
  - wc.nii / wc.nii.gz
  - whole_cerebellum.nii / whole_cerebellum.nii.gz
  - Centiloid_WC_VOI.nii / Centiloid_WC_VOI.nii.gz
  - CerebellumWholeMask.nii / CerebellumWholeMask.nii.gz


ALTERNATIVE LOCATION
--------------------

You can also store VOI files in a different location by setting the
environment variable:

    export CENTILOID_VOI_DIR=/path/to/your/voi/files


VERIFICATION
------------

After placing the files, run the software. The log should show:

    "Loaded official GAAIN VOI files: CTX=... WC=..."

If you see a warning about "procedural fallback masks", the files were
not found or could not be loaded.


REFERENCE
---------

Klunk WE et al. The Centiloid Project: Standardizing quantitative amyloid
plaque estimation by PET. Alzheimers Dement. 2015;11(1):1-15.

https://www.gaain.org/centiloid-project

================================================================================
