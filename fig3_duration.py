"""Manuscript Fig. 2: duration-selectivity curves of the persistence-gated LIF; d50 vs gate
duration c under two response windows.

Pulse-on window: a spike counts only if it occurs before pulse offset (the
assumption behind Eq. 8); predicted d50 = tau_m ln2 + c (Eq. 9).
Post-offset window: spikes after offset also count, as in a standard
strength-duration protocol; predicted d50 = tau_m ln(1 + e^{c/tau_m}) (Eq. 9b).
"""
from figstyle import *
from msmodels import first_spike_batch, d50_bootstrap
import json

TAU_M, R, VTH = 2.0, 100.0, 15.0
DT = 0.02
SIGMA, NTRIAL = 0.003, 500
CS = [0.5, 1.0, 2.0, 4.0, 8.0]
A = 2.0 * VTH / R                      # twice rheobase (nA)
TAIL = 6 * TAU_M                       # post-offset observation window
rc = TAU_M * np.log(2)


def pred_on(c):
    return rc + np.asarray(c)


def pred_post(c):
    return TAU_M * np.log(1 + np.exp(np.asarray(c) / TAU_M))


rng = np.random.default_rng(42)
cols = [C_PLUM, C_BLUE, '#2A7FA8', C_GREEN, '#8DC63F']

res = {}
for window, tail, pred in [('on', 0.0, pred_on), ('post', TAIL, pred_post)]:
    d50s, ses, curves = [], [], {}
    for c in CS:
        centre = float(pred(c))
        durs = np.round(np.arange(max(0.2, centre - 2.0), centre + 2.0 + 1e-9, 0.1), 3)
        p = np.array([np.mean(np.isfinite(first_spike_batch(
            A, d, tail, DT, TAU_M, R, VTH, c, SIGMA, NTRIAL, rng))) for d in durs])
        curves[c] = (durs, p)
        est, se = d50_bootstrap(durs, p, NTRIAL, rng=rng)
        d50s.append(est); ses.append(se)
        print('%-4s c=%4.1f  d50=%.3f +- %.3f  (prediction %.3f)'
              % (window, c, est, se, centre))
    d50s = np.array(d50s)
    slope, inter = np.polyfit(CS, d50s, 1)
    print('%-4s FIT slope=%.4f intercept=%.4f  max|d50-pred|=%.3f ms'
          % (window, slope, inter, np.max(np.abs(d50s - pred(CS)))))
    res[window] = dict(d50=d50s, se=np.array(ses), curves=curves,
                       slope=slope, intercept=inter)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.0),
                               gridspec_kw=dict(width_ratios=[1.15, 1], wspace=0.55))
for c, col in zip(CS, cols):
    d, p = res['on']['curves'][c]
    ax1.plot(d, p, 'o-', color=col, ms=3, lw=1.4,
             label='$c$ = %g ms' % c)
    d, p = res['post']['curves'][c]
    ax1.plot(d, p, 's--', color=col, ms=2.5, lw=1.0, mfc='white')
ax1.axhline(0.5, color=C_GRAY, ls=':', lw=0.9)
ax1.axvline(rc, color=C_GRAY, ls=':', lw=1.0)
ax1.text(rc - 0.1, 0.25, r'$\tau_m\ln 2$', fontsize=8.5, color=C_GRAY, ha='right')
ax1.plot([], [], 'k-o', ms=3, lw=1.4, label='pulse-on window')
ax1.plot([], [], 'k--s', ms=2.5, lw=1.0, mfc='white', label='post-offset window')
ax1.set_xlabel('Stimulus duration (ms)'); ax1.set_ylabel('P(spike)')
ax1.set_title(r'A. Detection vs duration ($\tau_m$ = 2 ms, $2\times$rheobase)',
              loc='left', fontsize=10.5, fontweight='bold')
ax1.legend(fontsize=7.8, frameon=False, loc='upper left', bbox_to_anchor=(1.0, 1.0))
ax1.set_ylim(-0.05, 1.08)

cc = np.linspace(0, 9, 100)
ax2.plot(cc, pred_on(cc), '-', color=C_RED, lw=1.5,
         label=r'Eq. 9: $\tau_m\ln 2 + c$')
ax2.plot(cc, pred_post(cc), '-', color=C_BLUE, lw=1.5,
         label=r'Eq. 9b: $\tau_m\ln(1+e^{c/\tau_m})$')
ax2.errorbar(CS, res['on']['d50'], yerr=res['on']['se'], fmt='o', color=C_RED,
             ms=6, capsize=2, label='simulated, pulse-on')
ax2.errorbar(CS, res['post']['d50'], yerr=res['post']['se'], fmt='s', color=C_BLUE,
             mfc='white', ms=6, capsize=2, label='simulated, post-offset')
ax2.set_xlabel(r'Gate duration $c$ (ms)'); ax2.set_ylabel(r'$d_{50}$ (ms)')
ax2.set_title(r'B. $d_{50}$ depends on the response window', loc='left',
              fontsize=10.5, fontweight='bold')
ax2.legend(fontsize=7.8, frameon=False, loc='upper left')

fig.savefig('fig3_duration_selectivity.pdf'); fig.savefig('fig3_duration_selectivity.png', dpi=150)
json.dump({'c': CS, 'rc': rc,
           'd50_on': list(res['on']['d50']), 'se_on': list(res['on']['se']),
           'slope_on': res['on']['slope'], 'intercept_on': res['on']['intercept'],
           'd50_post': list(res['post']['d50']), 'se_post': list(res['post']['se']),
           'pred_post': list(map(float, pred_post(CS)))},
          open('_out_fig3.json', 'w'), indent=1)
print('saved fig3_duration_selectivity.pdf')
