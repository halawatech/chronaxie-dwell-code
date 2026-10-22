"""Manuscript Fig. 3: can the dwell term be recovered from a strength-duration curve?

Parameter recovery for the persistence-gated model. Synthetic strength-duration
curves are generated from Eq. (8) (pulse-on window) or Eq. (8b) (post-offset
window) at up to 18 pulse durations (0.1-100 ms). Durations whose threshold
exceeds 50 x rheobase are treated as unmeasurable and omitted, as are pulse-on
durations d <= c, which have no finite threshold. Each threshold carries
independent multiplicative noise with coefficient of variation SIGMA (trial-to-
trial threshold variability and amplitude resolution). Rheobase b, tau_m and c
are all fitted (least squares on log threshold). The profile likelihood of c
(minimised over b in closed form and over tau_m on a fine grid) gives a 95%
interval (likelihood-ratio threshold 3.84, refined on a 0.002-ms grid when
narrow) and a one-sided test of c = 0 (threshold 2.71, appropriate at the
boundary, computed against the final minimum). 200 synthetic experiments per
condition; detection probabilities are reported with their binomial Monte
Carlo standard error, and interval coverage is reported rather than assumed.
The tau_m grid spans tau_true/3 to 3 tau_true: a benchmark design (the true
value is known), not a procedure for an experiment in which tau_m is unknown.
"""
from figstyle import *
from scipy.optimize import least_squares
import json
import sys

RNG = np.random.default_rng(42)
DURS = np.array([0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5, 7, 10, 15, 20,
                 30, 50, 100.0])
CAP = 50.0                               # largest measurable threshold / rheobase
CS_TRUE = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
SIGMAS = [0.01, 0.03]
TAUS = [15.0, 2.0]                       # physiological soma; model regime of Fig. 2
NSET = 200
C_COARSE = np.round(np.arange(0.0, 6.0001, 0.02), 4)


def shape_post(d, tau, c):
    return 1 + np.exp(c / tau) / np.expm1(d / tau)


def shape_on(d, tau, c):
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        return np.where(d > c, 1.0 / (-np.expm1(-(d - c) / tau)), np.inf)


def profile(logy, d, shape, tau_true, cgrid):
    taus = np.geomspace(tau_true / 3, tau_true * 3, 700)
    T, Cg, Dd = np.meshgrid(taus, cgrid, d, indexing='ij')
    s = shape(Dd, T, Cg)
    ok = np.all(np.isfinite(s), axis=2)
    r = logy[None, None, :] - np.log(np.where(np.isfinite(s), s, 1.0))
    r = r - r.mean(axis=2, keepdims=True)
    ssr = np.where(ok, (r ** 2).sum(axis=2), np.inf)
    return ssr.min(axis=0)


def analyse(logy, d, shape, tau, sg):
    prof = profile(logy, d, shape, tau, C_COARSE)
    best = prof.min()
    chi = (prof - best) / sg ** 2
    inside = C_COARSE[chi <= 3.84]
    lo, hi = inside.min(), inside.max()
    est = C_COARSE[int(np.argmin(prof))]
    if hi - lo < 0.2:                        # refine a narrow interval
        fine = np.arange(max(0.0, lo - 0.06), hi + 0.06, 0.002)
        pf = profile(logy, d, shape, tau, fine)
        best = min(best, pf.min())
        cf = (pf - best) / sg ** 2
        inside = fine[cf <= 3.84]
        lo, hi = inside.min(), inside.max()
        est = fine[int(np.argmin(pf))]
    # test of c = 0 against the final (possibly refined) minimum
    return est, lo, hi, bool((prof[0] - best) / sg ** 2 > 2.71)


# `python fig_identifiability.py --plot-only` redraws the figure from the saved JSON
PLOT_ONLY = '--plot-only' in sys.argv
results = json.load(open('_out_identifiability.json'))['results'] if PLOT_ONLY else {}
for tau in ([] if PLOT_ONLY else TAUS):
    for window, shape in [('post', shape_post), ('on', shape_on)]:
        for sg in SIGMAS:
            for c in CS_TRUE:
                y0 = shape(DURS, tau, c)
                keep = np.isfinite(y0) & (y0 <= CAP)
                d, y0 = DURS[keep], y0[keep]
                est, lo95, hi95, rej = [], [], [], []
                for _ in range(NSET):
                    e, lo, hi, rj = analyse(np.log(y0) + sg * RNG.standard_normal(len(d)),
                                            d, shape, tau, sg)
                    est.append(e); lo95.append(lo); hi95.append(hi); rej.append(rj)
                est, lo95, hi95 = map(np.array, (est, lo95, hi95))
                key = '%g|%s|%g|%g' % (tau, window, sg, c)
                results[key] = dict(
                    n_durations=int(len(d)), shortest=float(d.min()),
                    est_median=float(np.median(est)),
                    est_q10_q90=[float(np.quantile(est, 0.1)), float(np.quantile(est, 0.9))],
                    ci_width_median=float(np.median(hi95 - lo95)),
                    ci_upper_median=float(np.median(hi95)),
                    p_reject_c0=float(np.mean(rej)),
                    p_reject_se=float(np.sqrt(np.mean(rej) * (1 - np.mean(rej)) / NSET)),
                    coverage=float(np.mean((lo95 - 1e-9 <= c) & (c <= hi95 + 1e-9))))
                r = results[key]
                print('tau=%4.1f %-4s sigma=%.2f c=%.2f  n_d=%2d (from %.2f ms)  c_hat %.3f '
                      '[%.3f, %.3f]  CI width %.3f  P(reject c=0) %.2f  coverage %.2f'
                      % (tau, window, sg, c, r['n_durations'], r['shortest'], r['est_median'],
                         *r['est_q10_q90'], r['ci_width_median'], r['p_reject_c0'],
                         r['coverage']), flush=True)


# noise-free signal: deviation of the gate curve from the best no-gate (RC) fit
def rc_misfit(d, y):
    f = lambda p: np.log(p[0]) + np.log(shape_post(d, p[1], 0.0)) - np.log(y)
    r = least_squares(f, [1.0, d.mean()], bounds=([0, 0.01], [100, 300]))
    return 100 * r.fun


sig = {}
for window, shape in [('post', shape_post), ('on', shape_on)]:
    for c in (0.5, 1.0):
        y = shape(DURS, 15.0, c)
        keep = np.isfinite(y) & (y <= CAP)
        sig['%s|%g' % (window, c)] = (DURS[keep].tolist(), rc_misfit(DURS[keep], y[keep]).tolist())
        print('signal %-4s c=%.1f: max |deviation from best RC fit| = %.2f%%'
              % (window, c, np.max(np.abs(sig['%s|%g' % (window, c)][1]))))

# ---------------------------------------------------------------- figure
fig, axs = plt.subplots(1, 3, figsize=(12.6, 3.7), gridspec_kw=dict(wspace=0.33))
ax = axs[0]
for (window, c), col, st in [(('post', 0.5), C_BLUE, '-'), (('post', 1.0), C_BLUE, '--'),
                             (('on', 0.5), C_RED, '-'), (('on', 1.0), C_RED, '--')]:
    dd, rr = sig['%s|%g' % (window, c)]
    ax.plot(dd, np.maximum(np.abs(rr), 1e-3), st, marker='o', ms=3, color=col, lw=1.4,
            label='%s, $c$ = %g ms' % ('post-offset' if window == 'post' else 'pulse-on', c))
for s_ in SIGMAS:
    ax.axhline(100 * s_, color='#777777', ls=':', lw=1.0)
    ax.text(0.45, 100 * s_ * 1.12, '%d%% noise' % round(100 * s_), fontsize=7.4, color='#555555')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(0.01, 150)
ax.set_xlabel('Pulse duration (ms)'); ax.set_ylabel('|Gate curve $-$ best RC fit| (%)')
ax.set_title(r'A. Signal left by the dwell ($\tau_m$ = 15 ms)', loc='left', fontsize=10.5,
             fontweight='bold')
ax.legend(fontsize=7.2, frameon=False, loc='lower right')

for k, (metric, ylab, ttl) in enumerate([
        ('p_reject_c0', 'P(detect $c > 0$)', 'B. Detecting the dwell'),
        ('ci_width_median', 'Median 95% interval width (ms)', 'C. Precision of $\\hat c$')]):
    ax = axs[k + 1]
    for (tau, window, sg), col, st, mk in [((15.0, 'post', 0.01), C_BLUE, '-', 'o'),
                                           ((15.0, 'post', 0.03), C_BLUE, '--', 's'),
                                           ((15.0, 'on', 0.01), C_RED, '-', 'o'),
                                           ((15.0, 'on', 0.03), C_RED, '--', 's'),
                                           ((2.0, 'post', 0.03), C_GREEN, '-.', '^')]:
        cs = [c for c in CS_TRUE if not (metric == 'p_reject_c0' and False)]
        v = [results['%g|%s|%g|%g' % (tau, window, sg, c)][metric] for c in cs]
        err = ([results['%g|%s|%g|%g' % (tau, window, sg, c)].get('p_reject_se', 0) for c in cs]
               if metric == 'p_reject_c0' else None)
        ax.errorbar(cs, v, yerr=err, fmt=st, marker=mk, ms=4, color=col, lw=1.5, capsize=2,
                    label=r'%s, $\tau_m$ = %g ms, %d%%' % ('post-offset' if window == 'post'
                                                         else 'pulse-on', tau, round(100 * sg)))
    ax.set_xlabel('True dwell $c$ (ms)'); ax.set_ylabel(ylab)
    ax.set_title(ttl, loc='left', fontsize=10.5, fontweight='bold')
    if metric == 'p_reject_c0':
        ax.axhline(0.05, color='#999999', ls=':', lw=0.9)
        ax.set_ylim(-0.03, 1.05)
        handles, labels = ax.get_legend_handles_labels()
    else:
        ax.set_ylim(bottom=0)
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.645, 0.0), ncol=3,
           fontsize=7.8, frameon=False)

fig.savefig('fig_identifiability.pdf', bbox_inches='tight')
fig.savefig('fig_identifiability.png', dpi=150, bbox_inches='tight')
json.dump(dict(results=results, signal=sig, cap=CAP, nset=NSET, durations=DURS.tolist()),
          open('_out_identifiability.json', 'w'), indent=1)
print('saved fig_identifiability.pdf')
