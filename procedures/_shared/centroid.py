"""Centroid-fit primitives for procedures that need to locate a beam
spot on a 2-D detector frame.

Default fit is center-of-mass on an above-threshold ROI (robust,
fast, good enough for the alignment procedures' linear-slope use
case). Gaussian-fit upgrade lands here when a procedure asks for it.
"""

# TODO: implement center_of_mass(frame, threshold) -> (x, y)
# TODO: implement gaussian_2d_fit(frame, roi) -> (x, y, sigma_x, sigma_y, amplitude)
