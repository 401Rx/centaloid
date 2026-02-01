"""
Centaloid - PET-CT Centiloid Calculator for Beta-Amyloid Imaging.

Computes the Centiloid value from amyloid PET DICOM series by:
1. Loading and reconstructing the 3-D PET volume from DICOM slices.
2. Spatially normalising the volume to MNI-152 space (affine registration).
3. Extracting mean uptake in cortical target and reference (whole-cerebellum)
   regions defined by the standard Centiloid atlas ROIs.
4. Computing the SUVr (target / reference) and converting to the Centiloid
   scale using published tracer-specific linear equations.
"""

__version__ = "1.0.0"
