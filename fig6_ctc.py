"""Supplementary Fig. S8: gate duration under presynaptic phase concentration (feedforward
frequency-threshold recruitment). Three independent input realisations per
condition; rates are mean +- s.d. across realisations."""
from figstyle import *
from msmodels import lif_gate
import json

NPRE, R0, MOD = 200, 8.0, 0.9
TAU_SYN, QSPK = 3.0, 4e-14          # C per spike
TAU_M, R, VTH, T_REF = 10.0, 100.0, 8.0, 2.0
CS = [0.0, 0.5, 2.0, 5.0, 10.0, 30.0]
KAPPAS = [0, 1, 5, 20]
FOSC = [40.0, 20.0, 6.0]
SEEDS = [0, 1, 2]
T, DT, BURN = 5000.0, 0.1, 500.0
n = int(T / DT); t = np.arange(n) * DT


def presyn(f, kappa, rng):
    """Poisson units with von Mises-distributed phases; returns summed EPSC (nA)."""
    phi = rng.vonmises(0.0, kappa, NPRE) if kappa > 0 else \
        rng.uniform(-np.pi, np.pi, NPRE)
    counts = np.zeros(n)
    for i in range(NPRE):
        lam = R0 * (1 + MOD * np.cos(2 * np.pi * f * t * 1e-3 + phi[i]))
        lam = np.clip(lam, 0, None)
        p = lam * DT * 1e-3
        counts += (rng.random(n) < p).astype(float)
    # alpha kernel, unit area -> charge QSPK per spike
    kt = np.arange(0, 10 * TAU_SYN, DT)
    k = (kt / TAU_SYN) * np.exp(1 - kt / TAU_SYN)
    k = k / (k.sum() * DT * 1e-3)          # 1/s normalisation
    I = np.convolve(counts, k, mode='full')[:n] * QSPK * 1e9   # -> nA
    return I, phi


rates = {f: np.zeros((len(SEEDS), len(CS), len(KAPPAS))) for f in FOSC}
traces, vstats = {}, {}
TW = (BURN, BURN + 500.0)                 # example-trace window (post burn-in)
m_tw = (t >= TW[0]) & (t < TW[1])
for f in FOSC:
    for s_i, seed in enumerate(SEEDS):
        for j, kap in enumerate(KAPPAS):
            rng = np.random.default_rng(1000 * seed + 42 + int(f) * 17 + kap)
            I, phi = presyn(f, kap, rng)
            if seed == 0:
                _, vp = lif_gate(I, DT, TAU_M, R, VTH, 1e9, t_ref=T_REF)   # passive
                vp = vp[t >= BURN]
                vstats[(f, kap)] = (float(vp.mean()), float(np.percentile(vp, 99)))
            for i, c in enumerate(CS):
                sp, v = lif_gate(I, DT, TAU_M, R, VTH, c, t_ref=T_REF)
                sp_an = sp[sp >= BURN]
                rates[f][s_i, i, j] = len(sp_an) / ((T - BURN) * 1e-3)
                if seed == 0 and kap == 20:
                    sp_w = sp[(sp >= TW[0]) & (sp < TW[1])] - TW[0]
                    traces.setdefault(f, {})[c] = (v[m_tw], sp_w)
    mu = rates[f].mean(0)
    print('f=%2.0f Hz  rate(kappa=20): %s' % (f, ' '.join('%.1f' % r for r in mu[:, -1])))
    print('          rate(kappa=0) : %s' % ' '.join('%.1f' % r for r in mu[:, 0]))
    print('          passive V mean/p99 kappa=0: %.2f/%.2f  kappa=20: %.2f/%.2f mV'
          % (*vstats[(f, 0)], *vstats[(f, 20)]))

MU = {f: rates[f].mean(0) for f in FOSC}
SD = {f: rates[f].std(0) for f in FOSC}

fig = plt.figure(figsize=(11.5, 8.8))
gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.95], hspace=0.6, wspace=0.3)
vmax = max(MU[x].max() for x in FOSC)
for k, f in enumerate(FOSC):
    ax = fig.add_subplot(gs[0, k])
    M = MU[f]
    im = ax.imshow(M, origin='lower', aspect='auto', cmap='YlOrRd', vmin=0, vmax=vmax)
    ax.set_xticks(range(len(KAPPAS))); ax.set_xticklabels(KAPPAS)
    ax.set_yticks(range(len(CS))); ax.set_yticklabels(['%g' % c for c in CS])
    ax.set_xlabel(r'Phase concentration $\kappa$')
    if k == 0:
        ax.set_ylabel('Gate duration $c$ (ms)')
    lbl = {40.0: r'$\gamma$ (40 Hz)', 20.0: r'$\beta$ (20 Hz)', 6.0: r'$\theta$ (6 Hz)'}[f]
    ax.set_title('%s. %s' % ('ABC'[k], lbl), loc='left', fontsize=10.5, fontweight='bold')
    for i in range(len(CS)):
        for j in range(len(KAPPAS)):
            ax.text(j, i, '%.1f' % M[i, j], ha='center', va='center', fontsize=7.5,
                    color='white' if M[i, j] > 0.55 * vmax else '#333333')
fig.colorbar(im, ax=fig.axes[:3], fraction=0.02, pad=0.01, label='Postsyn. rate (Hz)')

ax = fig.add_subplot(gs[1, :])
xs = np.array([max(c, 0.2) for c in CS])      # plot c = 0 at 0.2 on the log axis
for f, col, mk in zip(FOSC, [C_BLUE, C_GREEN, C_AMBER], ['o', 's', '^']):
    ax.errorbar(xs, MU[f][:, -1], yerr=SD[f][:, -1], fmt=mk + '-', color=col, lw=2,
                ms=6, capsize=3,
                label={40.0: r'$\gamma$ 40 Hz', 20.0: r'$\beta$ 20 Hz', 6.0: r'$\theta$ 6 Hz'}[f])
ax.axhline(5.0, color=C_GRAY, ls='--', lw=1.2)
ax.text(30, 5.6, '5 Hz criterion', fontsize=8.5, color=C_GRAY, ha='right')
ax.set_xscale('log')
ax.set_xticks(xs); ax.set_xticklabels(['0 (std LIF)'] + ['%g' % c for c in CS[1:]])
ax.set_xlabel('Postsynaptic gate duration $c$ (ms)')
ax.set_ylabel('Firing rate (Hz)')
ax.set_title(r'D. Recruitment at $\kappa$ = 20 (mean $\pm$ s.d., 3 input realisations)',
             loc='left', fontsize=10.5, fontweight='bold')
ax.legend(fontsize=9, frameon=False)

ax = fig.add_subplot(gs[2, :])
off = 0
CS_TR = [0.0, 5.0, 30.0]                 # traces shown in E (all six are simulated)
cols = ['#222222', C_GREEN, C_RED]
tt = np.arange(m_tw.sum()) * DT
for f in FOSC:
    y0 = off
    for c, col in zip(CS_TR, cols):
        v, sp = traces[f][c]
        ax.plot(tt, v + off, color=col, lw=0.6)
        ax.axhline(VTH + off, color='#CCCCCC', ls=':', lw=0.6)
        for s in sp:
            ax.plot([s, s], [off + VTH + 1, off + VTH + 4], color='#222222', lw=0.7)
        off += 16
    ax.text(-15, 0.5 * (y0 + off - 16) + VTH,
            {40.0: r'$\gamma$', 20.0: r'$\beta$', 6.0: r'$\theta$'}[f],
            fontsize=12, ha='right', va='center')
ax.plot([505, 505], [0, 10], color='#222222', lw=1.5, clip_on=False)
ax.text(510, 5, '10 mV', fontsize=8, va='center')
ax.set_yticks([]); ax.set_xlabel('Time after burn-in (ms)')
ax.set_xlim(-30, 530)
ax.set_title(r'E. Example postsynaptic traces ($\kappa$ = 20); within each band, bottom '
             r'to top: $c$ = 0 (black), 5 (green), 30 ms (red)', loc='left', fontsize=10.5,
             fontweight='bold')

fig.savefig('fig_ctc_simulation.pdf', bbox_inches='tight')
fig.savefig('fig_ctc_simulation.png', dpi=150, bbox_inches='tight')
json.dump({'cs': CS, 'kappas': KAPPAS, 'seeds': SEEDS,
           'rate_mean': {str(f): MU[f].tolist() for f in FOSC},
           'rate_sd': {str(f): SD[f].tolist() for f in FOSC},
           'passive_v_mean_p99': {'%g_%d' % k: v for k, v in vstats.items()}},
          open('_out_fig6.json', 'w'), indent=1)
print('saved fig_ctc_simulation.pdf')
