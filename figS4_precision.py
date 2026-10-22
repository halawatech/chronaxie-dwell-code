"""Supplementary Fig. S7: gate duration sets spike latency, not jitter (strongly suprathreshold drive)."""
from figstyle import *
from msmodels import lif_gate
import json

TAU_M, R, VTH, T_REF, DT = 5.0, 100.0, 8.0, 2.0, 0.02
CS = [0.1, 0.3, 1.0, 3.0, 10.0]
SIGS = [0.005, 0.015, 0.050]        # nA  (5, 15, 50 pA)
AMP, DUR, NTRIAL = 0.16, 40.0, 600
COLS = {0.005: C_PLUM, 0.015: C_RED, 0.050: C_AMBER}

n = int(DUR / DT)
lat, jit = {}, {}
for sg in SIGS:
    m, s = [], []
    for c in CS:
        rng = np.random.default_rng(42 + int(c * 100) + int(sg * 1e4))
        L = []
        for _ in range(NTRIAL):
            I = AMP + sg * rng.standard_normal(n)
            sp, _ = lif_gate(I, DT, TAU_M, R, VTH, c, t_ref=T_REF)
            if len(sp):
                L.append(sp[0])
        L = np.array(L)
        m.append(L.mean()); s.append(L.std())
    lat[sg], jit[sg] = np.array(m), np.array(s)
    print('sigma=%5.3f nA  latency %s' % (sg, ' '.join('%.2f' % x for x in m)))
    print('               jitter  %s' % ' '.join('%.3f' % x for x in s))

# latency should scale as t* + c
tstar = -TAU_M * np.log(1 - VTH / (AMP * R))
print('predicted t* = %.3f ms ; latency - c = %s'
      % (tstar, ' '.join('%.2f' % (lat[SIGS[1]][i] - CS[i]) for i in range(len(CS)))))
sl, ic = np.polyfit(CS, lat[SIGS[1]], 1)
print('latency vs c: slope=%.3f intercept=%.3f' % (sl, ic))

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
ax = axes[0]
cc = np.linspace(0, 11, 50)
ax.plot(cc, cc + tstar, '--', color=C_GRAY, lw=1.4, label=r'$t^* + c$')
for sg in SIGS:
    ax.plot(CS, lat[sg], 'o-', color=COLS[sg], lw=1.8, ms=5,
            label=r'$\sigma$ = %g pA' % (sg * 1000))
ax.set_xlabel('Gate duration $c$ (ms)'); ax.set_ylabel('Mean spike latency (ms)')
ax.set_title('A. Latency scales linearly with $c$', loc='left',
             fontsize=10.5, fontweight='bold')
ax.legend(fontsize=8.5, frameon=False)

ax = axes[1]
for sg in SIGS:
    ax.plot(CS, jit[sg], 'o-', color=COLS[sg], lw=1.8, ms=5,
            label=r'$\sigma$ = %g pA' % (sg * 1000))
ax.set_xscale('log')
ax.set_xlabel('Gate duration $c$ (ms)'); ax.set_ylabel(r'Spike-time jitter $\sigma_t$ (ms)')
ax.set_ylim(0, max(max(j) for j in jit.values()) * 1.35)
ax.set_title('B. Jitter independent of $c$ (this regime)', loc='left',
             fontsize=10.5, fontweight='bold')
ax.legend(fontsize=8.5, frameon=False)

plt.tight_layout()
fig.savefig('fig_spike_precision.pdf'); fig.savefig('fig_spike_precision.png', dpi=150)
json.dump({'cs': CS, 'latency': {str(k): list(v) for k, v in lat.items()},
           'jitter': {str(k): list(v) for k, v in jit.items()},
           'tstar': tstar, 'slope': sl}, open('_out_figS4.json', 'w'), indent=1)
print('saved fig_spike_precision.pdf')
