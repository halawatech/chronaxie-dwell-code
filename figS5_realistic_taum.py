"""Supplementary Fig. S1: the decomposition across the membrane time constants examined here,
under both response windows.

Pulse-on window (spike must occur before pulse offset; the assumption behind
Eq. 8): d50 - tau_m ln2 = c (Eq. 9).
Post-offset window (spikes after offset also count): d50 - tau_m ln2 =
tau_m ln[(1 + e^{c/tau_m})/2] (Eq. 9b), which is ~c/2 when c << tau_m.
"""
from figstyle import *
from msmodels import first_spike_batch, d50_from_curve
import json

R, DT = 100.0, 0.05
NTRIAL, SIGMA = 400, 0.003
CS = [0.0, 0.5, 2.0, 5.0, 10.0]
REGIMES = [('Idealised', 2.0, 8.0), ('Intermediate', 8.0, 12.0),
           ('Physiological', 15.0, 15.0), ('Compact L4', 30.0, 15.0)]
COLS = [C_GRAY, C_PLUM, C_BLUE, C_GREEN, C_AMBER]

fig, axes = plt.subplots(2, 4, figsize=(15, 7.0))
out = {}
for k, (name, tau_m, vth) in enumerate(REGIMES):
    rc = tau_m * np.log(2)
    A = 2.0 * vth / R
    pred = {'on': lambda c: rc + c,
            'post': lambda c: tau_m * np.log(1 + np.exp(c / tau_m))}
    d50 = {'on': [], 'post': []}
    ax = axes[0, k]
    rng = np.random.default_rng(7 + 991 * k)
    for c, col in zip(CS, COLS):
        for window, tail, ls, mk in [('on', 0.0, '-', 'o'), ('post', 6 * tau_m, '--', 's')]:
            centre = pred[window](c)
            durs = np.round(np.arange(max(0.2, centre - 2.5), centre + 2.5, 0.15), 3)
            p = np.array([np.mean(np.isfinite(first_spike_batch(
                A, d, tail, DT, tau_m, R, vth, c, SIGMA, NTRIAL, rng))) for d in durs])
            ax.plot(durs, p, ls, marker=mk, color=col, ms=2.2, lw=1.1,
                    mfc=(col if window == 'on' else 'white'),
                    label=('$c$ = %g ms' % c) if window == 'on' else None)
            d50[window].append(d50_from_curve(durs, p))
    ax.axvline(rc, color=C_GRAY, ls=':', lw=1.1)
    ax.set_xlabel('Stim. duration (ms)')
    if k == 0:
        ax.set_ylabel('P(spike)')
        ax.plot([], [], 'k-o', ms=2.2, lw=1.1, label='pulse-on')
        ax.plot([], [], 'k--s', ms=2.2, lw=1.1, mfc='white', label='post-offset')
        ax.legend(fontsize=7, frameon=False, loc='center right')
    ax.set_title(r'%s. %s ($\tau_m$=%g, $V_\theta$=%g)' % ('ABCD'[k], name, tau_m, vth),
                 loc='left', fontsize=9.5, fontweight='bold')
    ax.set_ylim(-0.05, 1.08)

    r_on = np.array(d50['on']) - rc
    r_post = np.array(d50['post']) - rc
    sl_on, ic_on = np.polyfit(CS, r_on, 1)
    sl_post, ic_post = np.polyfit(CS, r_post, 1)
    pp = np.array([pred['post'](c) for c in CS]) - rc
    out[name] = dict(tau_m=tau_m, vth=vth, rc=rc,
                     d50_on=list(map(float, d50['on'])), d50_post=list(map(float, d50['post'])),
                     slope_on=float(sl_on), intercept_on=float(ic_on),
                     slope_post=float(sl_post), intercept_post=float(ic_post),
                     max_err_on=float(np.max(np.abs(r_on - np.array(CS)))),
                     max_err_post=float(np.max(np.abs(r_post - pp))))
    print('%-14s tau_m=%4.1f  pulse-on slope=%.3f int=%+.3f maxerr=%.3f | '
          'post-offset slope=%.3f int=%+.3f maxerr vs Eq10b=%.3f'
          % (name, tau_m, sl_on, ic_on, out[name]['max_err_on'],
             sl_post, ic_post, out[name]['max_err_post']))
    ax = axes[1, k]
    cc = np.linspace(0, 11, 60)
    ax.plot(cc, cc, '-', color=C_RED, lw=1.4, label='Eq. 9 (pulse-on)')
    ax.plot(cc, tau_m * np.log(1 + np.exp(cc / tau_m)) - rc, '-', color=C_BLUE, lw=1.4,
            label='Eq. 9b (post-offset)')
    ax.plot(CS, r_on, 'o', color=C_RED, ms=6)
    ax.plot(CS, r_post, 's', color=C_BLUE, mfc='white', ms=6)
    ax.set_xlabel('Gate duration $c$ (ms)')
    if k == 0:
        ax.set_ylabel(r'$d_{50}-\tau_m\ln 2$ (ms)')
        ax.legend(fontsize=7.5, frameon=False, loc='upper left')
    ax.set_title('%s. Gate residual (%s)' % ('EFGH'[k], name), loc='left',
                 fontsize=9.5, fontweight='bold')

plt.tight_layout()
fig.savefig('fig_realistic_taum.pdf'); fig.savefig('fig_realistic_taum.png', dpi=150)
json.dump(out, open('_out_figS5.json', 'w'), indent=1)
print('saved fig_realistic_taum.pdf')
