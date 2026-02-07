================================================================================
                      CENTILOID VOI FILES
================================================================================

This directory contains the Volume of Interest (VOI) NIfTI files used for
Centiloid quantification. These files define the standardized brain regions
in MNI-152 space.


INCLUDED FILES
--------------

voi_ctx_2mm.nii
    Cortical target (CTX) VOI mask from GAAIN Centiloid Project.
    This is the standard cortical composite region derived from the
    AD-minus-YC PiB difference images (thresholded at 1.05 SUVr).
    Source: https://github.com/MahnazShekari/Centiloid-pipeline
    Original: GAAIN_crtx_2mm.nii

voi_wc_2mm.nii
    Whole cerebellum (WC) reference VOI mask.
    Anatomically-defined ellipsoidal mask covering the cerebellum,
    constrained to not overlap with CTX.

CL_Composite.nii
    AAL-based composite atlas (for reference only).
    Contains multiple labeled brain regions.

GAAIN_crtx_2mm.nii
    Original GAAIN cortex mask file (backup copy).


SPECIFICATIONS
--------------

- Space: MNI-152 (SPM8 normalization)
- Resolution: 2mm isotropic
- Dimensions: 91 x 109 x 91 voxels
- Format: NIfTI-1


OFFICIAL GAAIN FILES
--------------------

For the most accurate clinical results, you can download the official
GAAIN Centiloid VOI files from:

    https://www.gaain.org/centiloid-project

Download "Centiloid_Std_VOI.zip" and extract the files. Place them in
this directory with the names:
    - voi_ctx_2mm.nii (or voi_ctx_2mm.nii.gz)
    - voi_wc_2mm.nii (or voi_wc_2mm.nii.gz)

The software will automatically use any official files placed here.


ALTERNATIVE LOCATION
--------------------

You can store VOI files in a different location by setting:

    export CENTILOID_VOI_DIR=/path/to/your/voi/files


REFERENCE
---------

Klunk WE et al. The Centiloid Project: Standardizing quantitative amyloid
plaque estimation by PET. Alzheimers Dement. 2015;11(1):1-15.

https://www.gaain.org/centiloid-project

================================================================================
