"""
Supplementary Fig. S6: published chronaxie measurements vs the model's predicted ranges.

Measured values, clinical settings and model predictions are shown in separate
blocks and styles. Sources:
  Ranck (1975): review ranges for large/small myelinated axons and for cell
      bodies and dendrites (extracellular stimulation).
  Nowak & Bullier (1998, Part I; rat visual cortex slices in vitro):
      direct cell-body chronaxie by intracellular injection, n = 9,
      mean 15.0 ms, range 6.8-31.3 ms (membrane time constant 7.0 +- 2.5 ms);
      axonal/orthodromic chronaxie ~40x smaller than the cell-body value.
  Grill (2015): DBS pulse widths.
"""
from figstyle import *
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

C_SOMA_PRED = '#8B5A2B'    # brown: predicted somatic
C_CLIN = '#555555'         # grey: clinical

# (label, low, high, source, category, point)  point = marked mean or None
data = [
    ("Large myelinated axons",       0.030, 0.200, "Ranck 1975", "axon", None),
    ("Small myelinated axons",       0.200, 0.700, "Ranck 1975", "axon", None),
    ("Axonal / orthodromic, rat V1 slice", None, None, "Nowak & Bullier 1998",
     "axon", 15.0 / 40),
    ("Cell bodies and dendrites",    1.000, 10.00, "Ranck 1975", "soma", None),
    ("Cell body, intracellular, rat V1 slice (n = 9)", 6.825, 31.29,
     "Nowak & Bullier 1998", "soma", 15.0),
    ("DBS pulse widths",             0.060, 0.150, "Grill 2015", "clinical", None),
]

# c_AIS: one illustrative range for all classes, bounded above by axonal chronaxie
predictions_ais = [("All classes (not laminar-resolved)", 0.1, 0.5)]
# t_ch^soma = tau_m ln2 + c_AIS (Eq. 9), tau_m ranges from Sections 8.1-8.2
predictions_soma = [
    ("FS interneuron",       5.6, 6.0),
    ("L2/3 pyr",             7.0, 10.9),
    ("L5 pyr",              10.5, 14.4),
    ("L4 spiny stellate",   21.8, 22.2),
]

cat_color = {'axon': C_BLUE, 'soma': C_RED, 'clinical': C_CLIN}

fig, ax = plt.subplots(figsize=(11, 6.6))
y = 0
lit_y = []
for (label, lo, hi, src, cat, pt) in data:
    col = cat_color[cat]
    if lo is not None:
        ax.plot([lo, hi], [y, y], color=col, linewidth=5, solid_capstyle='round', alpha=0.85)
        right = hi
    else:
        right = pt
    if pt is not None:
        ax.plot(pt, y, 'D', color=col, markersize=8, markeredgecolor='white',
                markeredgewidth=1.2, zorder=5)
    ax.text(0.022, y, label, fontsize=8.8, ha='right', va='center')
    ax.text(right * 1.18, y, src, fontsize=7.5, ha='left', va='center',
            color=C_GRAY, style='italic')
    lit_y.append(y)
    y += 1
y_lit_end = y - 1
y += 1

y_ais = []
for (label, lo, hi) in predictions_ais:
    ax.plot([lo, hi], [y, y], color=C_AMBER, linewidth=4, solid_capstyle='butt', alpha=0.9)
    ax.plot([lo, hi], [y, y], 's', color=C_AMBER, markersize=7, markerfacecolor='white',
            markeredgewidth=1.5)
    ax.text(0.022, y, '$c_{\\rm AIS}$: ' + label, fontsize=8.8, ha='right', va='center')
    y_ais.append(y)
    y += 1
y_ais_end = y - 1
y += 1

y_soma = []
for (label, lo, hi) in predictions_soma:
    ax.plot([lo, hi], [y, y], color=C_SOMA_PRED, linewidth=4, solid_capstyle='butt', alpha=0.9)
    ax.plot([lo, hi], [y, y], 's', color=C_SOMA_PRED, markersize=7, markerfacecolor='white',
            markeredgewidth=1.5)
    ax.text(0.022, y, '$t_{\\rm ch}^{\\rm soma}$: ' + label, fontsize=8.8, ha='right', va='center')
    y_soma.append(y)
    y += 1
y_max = y

ax.axhline(y_lit_end + 1.0, color=C_GRAY, linestyle=':', linewidth=0.6)
ax.axhline(y_ais_end + 1.0, color=C_GRAY, linestyle=':', linewidth=0.6)

for yy, lab in [(np.mean(lit_y), 'Measured /\nclinical'), (np.mean(y_ais), 'Model\n(assumed)'),
                (np.mean(y_soma), 'Model\n(illustrative)')]:
    ax.text(115, yy, lab, fontsize=9.5, fontweight='bold', ha='left', va='center',
            color='#444444')

legend_elems = [
    Patch(facecolor=C_BLUE, label='Measured: axonal'),
    Patch(facecolor=C_RED, label='Measured: cell body / dendrite'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='#444444', markersize=7,
           label='Mean (Nowak & Bullier)'),
    Patch(facecolor=C_CLIN, label='Clinical (DBS)'),
    Patch(facecolor=C_AMBER, label='Model: $c_{\\rm AIS}$ (assumed)'),
    Patch(facecolor=C_SOMA_PRED, label='Model: $t_{\\rm ch}^{\\rm soma}$ (Eq. 9; illustrative)'),
]
ax.legend(handles=legend_elems, loc='lower right', bbox_to_anchor=(1.0, 0.02),
          fontsize=8, frameon=True, facecolor='white', edgecolor='#CCCCCC')

ax.set_xscale('log')
ax.set_xlim(0.025, 100)
ax.set_ylim(-0.8, y_max - 0.3)
ax.set_yticks([])
ax.set_xlabel('Chronaxie (ms)', fontsize=11)
ax.set_title('Published chronaxie measurements vs. conditional model illustrations',
             fontsize=12, pad=10)
for s in ('top', 'right', 'left'):
    ax.spines[s].set_visible(False)
ax.tick_params(axis='y', length=0)
ax.grid(True, axis='x', linestyle=':', linewidth=0.4, alpha=0.5, which='both')
ax.set_axisbelow(True)

plt.tight_layout()
fig.savefig('fig_empirical_overlay.pdf')
fig.savefig('fig_empirical_overlay.png', dpi=150)
print('Fig S6 (empirical overlay) saved -> fig_empirical_overlay.pdf')
