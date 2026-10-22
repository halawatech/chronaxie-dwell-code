"""Supplementary Fig. S5: non-resonant gated neurons probed one frequency at a time.

Each gate duration c is driven by a single sinusoid (plus white current noise)
at each probe frequency in turn, so no probe frequency is a harmonic of
another component of the same input. Reported per (c, f): mean rate, the
absolute rate-modulation amplitude at f (spikes/s; not normalised), and
vector strength (phase locking of spikes to the drive).
"""
from figstyle import *
from msmodels import lif_gate
import json

TAU_M, R, VTH, T_REF, DT = 3.0, 100.0, 2.0, 2.0, 0.05
CS = [0.0, 0.5, 2.0, 4.0, 8.0, 20.0, 40.0]
FREQS = [2.0, 8.0, 20.0, 50.0]
AMP, SIG = 0.04, 0.004
T, BURN = 6500.0, 500.0
n = int(T / DT); t = np.arange(n) * DT


def analyse(sp, f):
    ncyc = int(np.floor((T - BURN) * 1e-3 * f))
    win = ncyc / f * 1e3
    s = sp[(sp >= BURN) & (sp < BURN + win)]
    rate = len(s) / (win * 1e-3)
    if len(s) < 2:
        return rate, 0.0, 0.0
    ph = 2 * np.pi * f * s * 1e-3
    z = np.exp(1j * ph).sum()
    mod = 2 * np.abs(z) / (win * 1e-3)       # modulation amplitude (spikes/s)
    vs = np.abs(z) / len(s)                  # vector strength
    return rate, float(mod), float(vs)


RATE = np.zeros((len(CS), len(FREQS)))
MOD = np.zeros_like(RATE)
VS = np.zeros_like(RATE)
example = None
for j, f in enumerate(FREQS):
    rng = np.random.default_rng(42 + j)
    I = AMP * np.sin(2 * np.pi * f * t * 1e-3) + SIG * rng.standard_normal(n)
    for i, c in enumerate(CS):
        sp, v = lif_gate(I, DT, TAU_M, R, VTH, c, t_ref=T_REF)
        RATE[i, j], MOD[i, j], VS[i, j] = analyse(sp, f)
        if f == 20.0 and c in (0.5, 8.0, 20.0):
            example = example or {}
            example[c] = (v, sp)
    print('f=%4.0f Hz  rate: %s' % (f, ' '.join('%6.1f' % x for x in RATE[:, j])))
    print('          mod : %s' % ' '.join('%6.1f' % x for x in MOD[:, j]))
    print('          VS  : %s' % ' '.join('%6.2f' % x for x in VS[:, j]))

best = [FREQS[int(np.argmax(MOD[i]))] if MOD[i].max() > 0 else None for i in range(len(CS))]
print('frequency of largest modulation per c:', dict(zip(CS, best)))

fig = plt.figure(figsize=(12, 7.2))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.1], hspace=0.5, wspace=0.45)

ax = fig.add_subplot(gs[0, :])
w = (t >= BURN) & (t < BURN + 300)
off = 0
for c, col in zip((0.5, 8.0, 20.0), (C_PLUM, C_GREEN, C_RED)):
    v, sp = example[c]
    ax.plot(t[w] - BURN, v[w] + off, color=col, lw=0.8, label='$c$ = %g ms' % c)
    ax.axhline(VTH + off, color='#BBBBBB', ls=':', lw=0.7)
    for s in sp[(sp >= BURN) & (sp < BURN + 300)]:
        ax.plot([s - BURN] * 2, [off + VTH + 0.4, off + VTH + 1.6], color='#222222', lw=0.8)
    off += 7
ax.plot([304, 304], [0, 5], color='#222222', lw=1.5, clip_on=False)
ax.text(307, 2.5, '5 mV', fontsize=8, va='center')
ax.set_xlim(0, 318)
ax.set_yticks([]); ax.set_xlabel('Time (ms)')
ax.set_title('A. 20-Hz probe, same input for each $c$ (traces offset by 7 mV)', loc='left',
             fontsize=10.5, fontweight='bold')
ax.legend(fontsize=8, frameon=False, loc='lower right', bbox_to_anchor=(1.0, 1.0), ncol=3)

for k, (M, ttl, cmap, fmt) in enumerate([(RATE, 'B. Mean rate (spikes/s)', 'YlOrRd', '%.0f'),
                                         (MOD, 'C. Modulation at $f$ (spikes/s)', 'YlOrRd', '%.0f'),
                                         (VS, 'D. Vector strength', 'Blues', '%.2f')]):
    ax = fig.add_subplot(gs[1, k])
    im = ax.imshow(M, origin='lower', aspect='auto', cmap=cmap,
                   vmin=0, vmax=(1 if M is VS else M.max()))
    ax.set_xticks(range(len(FREQS))); ax.set_xticklabels(['%g' % f for f in FREQS])
    ax.set_yticks(range(len(CS))); ax.set_yticklabels(['%g' % c for c in CS])
    ax.set_xlabel('Probe frequency (Hz)')
    if k == 0:
        ax.set_ylabel('Gate duration $c$ (ms)')
    for i in range(len(CS)):
        for j in range(len(FREQS)):
            ax.text(j, i, fmt % M[i, j], ha='center', va='center', fontsize=7.5,
                    color='white' if M[i, j] > 0.6 * (1 if M is VS else M.max()) else '#333333')
    ax.set_title(ttl, loc='left', fontsize=10.5, fontweight='bold')

fig.savefig('fig_population.pdf', bbox_inches='tight')
fig.savefig('fig_population.png', dpi=150, bbox_inches='tight')
json.dump({'cs': CS, 'freqs': FREQS, 'rate': RATE.tolist(), 'mod': MOD.tolist(),
           'vs': VS.tolist()}, open('_out_figS3.json', 'w'), indent=1)
print('saved fig_population.pdf')
