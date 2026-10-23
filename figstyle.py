"""
Shared figure style for the chronaxie manuscript figures.

NOTE: This file was reconstructed (the original was not present in the
folder) to provide the symbols imported via `from figstyle import *`
by fig7_empirical.py and figS5_realistic_taum.py, namely:
  - np   (numpy)
  - plt  (matplotlib.pyplot)
  - colour constants C_BLUE, C_RED, C_GREEN, C_PLUM, C_AMBER, C_GRAY
Colours are chosen to match the semantic names; exact hexes may differ
from the original style sheet. The file sets appearance only and computes
nothing; every figure in the submitted manuscript was generated with this
version.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless; we only write files
import matplotlib.pyplot as plt

# --- Colour palette (publication-friendly, colour-blind aware) ---
C_BLUE  = "#2E6FB0"   # axonal literature
C_RED   = "#C0392B"   # somatic literature
C_GREEN = "#2E8B57"   # dendritic literature
C_PLUM  = "#7D3C98"   # clinical / predicted soma
C_AMBER = "#E08A1E"   # predicted AIS
C_GRAY  = "#777777"   # annotations / sources

# --- Global rcParams for a clean look ---
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 10,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "text.color": "#222222",
    "axes.labelcolor": "#222222",
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "pdf.fonttype": 42,   # embed TrueType (editable text in PDF)
    "ps.fonttype": 42,
})

__all__ = [
    "np", "plt",
    "C_BLUE", "C_RED", "C_GREEN", "C_PLUM", "C_AMBER", "C_GRAY",
]
