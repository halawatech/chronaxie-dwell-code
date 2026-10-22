"""Manuscript Fig. S1: stimulation-energy optimum at chronaxie (Weiss form, normalised c = 1)."""
from figstyle import *

c = 1.0
d = np.linspace(0.05, 5, 800)
I = 1 + c / d                 # threshold current / rheobase
E = I ** 2 * d                # threshold energy (arb.)
Q = I * d                     # threshold charge
P = I ** 2                    # threshold power

fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

ax = axes[0]
ax.plot(d, I, color=C_BLUE, lw=2)
ax.axvline(c, color=C_RED, ls='--', lw=1.5, label='Chronaxie ($d = c$)')
ax.axhline(1, color=C_GRAY, ls=':', lw=1.0)
ax.axhline(2, color=C_GRAY, ls=':', lw=1.0)
ax.text(4.9, 1.06, 'Rheobase', ha='right', fontsize=8.5, color=C_GRAY)
ax.text(4.9, 2.06, r'$2\times$rheobase', ha='right', fontsize=8.5, color=C_GRAY)
ax.set_ylim(0, 8); ax.set_xlim(0, 5)
ax.set_xlabel('Pulse duration (norm.)'); ax.set_ylabel('Threshold current (norm.)')
ax.set_title('A. Strength–duration', loc='left', fontsize=11, fontweight='bold')
ax.legend(fontsize=8.5, frameon=False)

ax = axes[1]
ax.plot(d, E, color=C_GREEN, lw=2)
ax.plot(c, 4 * c, 'o', color=C_RED, ms=8)
ax.axvline(c, color='#444444', ls='--', lw=1.2)
ax.annotate('Min at $d = c$\n$E_{\\min}=4I_{\\rm rh}^2Rc$', xy=(c, 4),
            xytext=(1.7, 8.5), fontsize=9, color=C_RED,
            arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.2))
ax.set_ylim(0, 20); ax.set_xlim(0, 5)
ax.set_xlabel('Pulse duration (norm.)'); ax.set_ylabel('Threshold energy (norm.)')
ax.set_title('B. Stimulation-energy minimum', loc='left', fontsize=11, fontweight='bold')

ax = axes[2]
ax.plot(d, Q, color=C_BLUE, lw=2, label='Charge')
ax.set_xlabel('Pulse duration (norm.)')
ax.set_ylabel('Threshold charge (norm.)', color=C_BLUE)
ax.tick_params(axis='y', labelcolor=C_BLUE)
ax.set_xlim(0, 5); ax.set_ylim(0, 7)
ax2 = ax.twinx()
ax2.plot(d, P, color=C_RED, lw=2, label='Power')
ax2.set_ylabel('Threshold power (norm.)', color=C_RED)
ax2.tick_params(axis='y', labelcolor=C_RED)
ax2.set_ylim(0, 130)
ax.axvline(c, color='#444444', ls='--', lw=1.2)
ax.set_title('C. Charge vs. power tradeoff', loc='left', fontsize=11, fontweight='bold')
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8.5, frameon=False, loc='center right')

plt.tight_layout()
fig.savefig('fig2_energy_optimality.pdf'); fig.savefig('fig2_energy_optimality.png', dpi=150)
print('energy ratio at d/c=2 : %.4f' % (0.25 * (2 + 2 + 0.5)))
print('energy ratio at d/c=3 : %.4f' % (0.25 * (3 + 2 + 1 / 3)))
print('energy ratio at d/c=10: %.4f' % (0.25 * (10 + 2 + 0.1)))
print('saved fig2_energy_optimality.pdf')
