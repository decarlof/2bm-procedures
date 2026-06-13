"""B-station slit composite helpers.

The four B-station slit blades are addressed individually at the
motor level (``2bma:m9``/``m10`` vertical pair, ``2bma:m11``/``m12``
horizontal pair); the procedures want the higher-level Size and
Center handles that ``2slit.adl`` exposes.

Note the **horizontal-blade label flip** documented in
`2bm-docs/manual/item_020.rst` (B-station Slits block): the on-screen
"B slit Inb" / "B slit Outb" labels are mirrored with respect to
the physical inboard / outboard convention because the detector
image is left-right flipped. Helpers here follow the **physical**
convention, not the on-screen labels.
"""

# TODO: set_horizontal_aperture(size_mm, centre_mm=None) writing to
# the relevant composite PVs once the names are pinned in item_020.
# TODO: equivalent for vertical aperture.
# TODO: read-back helper that returns (size, centre) for verification.
