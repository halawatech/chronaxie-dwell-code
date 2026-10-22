"""Manuscript Fig. 1: HH strength-duration curves, and how far the persistence-gate
reduction describes them.

  A: HH thresholds (22 durations, 0.05-50 ms) with the post-offset gate equation
     (Eq. 8b) fitted to each sodium-conductance variant.
  B: chronaxie vs two RC predictions: leak-only (tau = C/gL) and resting
     membrane (tau_rest = C/g_rest, all conductances at their resting values).
  C: fit residuals, standard gNa: RC (Lapicque), Weiss, exponential
     integrate-and-fire (EIF), gate (b, tau_m, c free), gate with tau_m = tau_rest.
  D: held-out test: first-spike latency to 60-ms steps vs the gate prediction
     t*(A) + c from the parameters fitted in A.
Spikes after pulse offset count (V reaches 0 mV within 30 ms of offset); each
variant starts from its own resting equilibrium; forward Euler, dt = 0.01 ms;
thresholds by bisection to a relative tolerance of 2e-4.
"""
from figstyle import *
from msmodels import (HH, hh_rest_state, hh_resting_tau, hh_first_spike,
                      hh_thresholds)
from scipy.optimize import least_squares, brentq
from matplotlib.ticker import NullFormatter
from scipy.integrate import quad
import json

DURS = np.geomspace(0.05, 50.0, 22)
GNAS = [80.0, 120.0, 160.0]
LABELS = {80.0: r'Low $g_{\rm Na}$ (80)', 120.0: r'Std $g_{\rm Na}$ (120)',
          160.0: r'High $g_{\rm Na}$ (160)'}
COLS = {80.0: C_PLUM, 120.0: C_RED, 160.0: C_AMBER}
K_LAT = np.array([1.1, 1.25, 1.5, 2.0, 3.0, 5.0, 8.0])

tau_leak = HH['C'] / HH['gL']          # 3.33 ms
rc_leak = tau_leak * np.log(2)         # 2.31 ms


# ---- strength-duration models (threshold / b as a function of duration) ----
def gate_post(d, tau, c):                 # Eq. (8b)
    return 1 + np.exp(c / tau) / np.expm1(d / tau)


def rc_curve(d, tau):                     # Eq. (3)
    return 1 / (-np.expm1(-d / tau))


def weiss(d, tch):
    return 1 + tch / d


# EIF threshold curve: d_min(a) is computed on a grid that is uniform in
# log(a - 1), i.e. dense near rheobase where d_min changes steeply, and inverted
# by interpolation in (log d, log(a - 1)); eif_direct() checks it by root solving.
X_GRID = np.linspace(np.log(1e-9), np.log(3e4), 400)


def eif_dmin(a, tau, dl):
    """EIF, tau du/dt = -u + dl exp((u - 1 - dl)/dl) + a (rheobase a = 1):
    minimum pulse duration at amplitude a when spikes after offset count, i.e.
    the time to reach the unstable fixed point of the unforced system."""
    f0 = lambda u: -u + dl * np.exp(np.minimum((u - 1 - dl) / dl, 700))
    us = brentq(f0, 1 + dl, 1 + 61 * dl + 60)
    ur = brentq(f0, -1, 1 + dl)
    return quad(lambda u: tau / (f0(u) + a), ur, us, limit=400)[0]


def eif_curve(d, tau, dl):
    dm = np.array([eif_dmin(1 + np.exp(x), tau, dl) for x in X_GRID])
    return 1 + np.exp(np.interp(np.log(d), np.log(dm[::-1]), X_GRID[::-1]))


def eif_direct(d, tau, dl):
    """Threshold / b by direct root solving of d_min(a) = d (check of eif_curve)."""
    out = []
    for dd in np.atleast_1d(d):
        g = lambda x: eif_dmin(1 + np.exp(x), tau, dl) - dd
        out.append(1 + np.exp(brentq(g, np.log(1e-12), np.log(3e5), xtol=1e-10)))
    return np.array(out)


def fit(model, y, p0, lb, ub, durs=DURS):
    ly = np.log(y)
    res = least_squares(lambda p: np.log(p[0]) + np.log(model(durs, *p[1:])) - ly, p0,
                        bounds=(lb, ub))
    return res.x, res.fun


def fit_multi(model, y, starts, lb, ub):
    """Best of several starting points (smallest sum of squared residuals)."""
    best = None
    for p0 in starts:
        p, r = fit(model, y, p0, lb, ub)
        if best is None or np.sum(r ** 2) < np.sum(best[1] ** 2):
            best = (p, r)
    return best


def small_pulse_tau(g, amp=0.2, dt=0.005, window=3.0):
    """Time constant of a single exponential fitted to the 0-3-ms response to
    +/- 0.2 uA/cm^2 steps (mean of the two)."""
    V0 = hh_rest_state(g)[0]
    _, rec = hh_first_spike(np.array([amp, -amp]), 20.0, g, dt=dt, tail=0.0, record=True)
    tt = (np.arange(rec.shape[0]) + 1) * dt
    sel = tt <= window
    taus = []
    for k in range(2):
        y = rec[sel, k] - V0
        r = least_squares(lambda p: p[0] * (1 - np.exp(-tt[sel] / p[1])) - y, [y[-1], 1.5])
        taus.append(r.x[1])
    return float(np.mean(taus))


out = {}
for g in GNAS:
    th = hh_thresholds(DURS, g)
    rheo_all = hh_thresholds(np.array([50.0, 100.0, 200.0]), g)
    rheo = float(rheo_all[0])
    tau_rest, tau_sp = hh_resting_tau(g), small_pulse_tau(g)
    y = th / rheo
    fits = {
        'RC': fit(rc_curve, th, [rheo, 2.0], [0, 0.05], [100, 50]),
        'Weiss': fit(weiss, th, [rheo, 1.5], [0, 0.01], [100, 50]),
        'EIF': fit_multi(eif_curve, th, [[rheo, 2.0, 0.3], [rheo, 2.0, 0.05], [rheo, 1.5, 1.0]],
                         [0, 0.05, 0.01], [100, 50, 5]),
        'Gate': fit(gate_post, th, [rheo, 2.0, 0.3], [0, 0.05, 0.0], [100, 50, 20]),
        'Gate_tau_rest': fit(lambda d, c: gate_post(d, tau_rest, c), th, [rheo, 0.3],
                             [0, 0.0], [100, 20]),
        'Gate_tau_pulse': fit(lambda d, c: gate_post(d, tau_sp, c), th, [rheo, 0.3],
                              [0, 0.0], [100, 20]),
    }
    b_g, tau_g, c_g = fits['Gate'][0]
    pe = fits['EIF'][0]
    eif_err = float(np.max(np.abs(np.log(eif_curve(DURS, pe[1], pe[2]))
                                  - np.log(eif_direct(DURS, pe[1], pe[2])))))
    # convergence: full curve and gate fit at half the time step
    th_half = hh_thresholds(DURS, g, dt=0.005)
    gate_half = fit(gate_post, th_half, [rheo, 2.0, 0.3], [0, 0.05, 0.0], [100, 50, 20])
    lat = hh_first_spike(K_LAT * rheo, 60.0, g, tail=0.0)
    lat_pred = -tau_g * np.log(1 - b_g / (K_LAT * rheo)) + c_g
    out[g] = dict(V0=hh_rest_state(g)[0], rheo=rheo, rheo_50_100_200=rheo_all.tolist(),
                  tau_rest=tau_rest, tau_small_pulse=tau_sp, th=th.tolist(),
                  fits={k: dict(p=list(map(float, p)),
                                rms_pct=float(100 * np.sqrt(np.mean(r ** 2))),
                                resid=list(map(float, r))) for k, (p, r) in fits.items()},
                  lat_K=K_LAT.tolist(), lat=lat.tolist(), lat_pred=lat_pred.tolist(),
                  eif_interp_vs_direct_max_log=eif_err,
                  dt_half=dict(th=th_half.tolist(), gate_p=list(map(float, gate_half[0])),
                               gate_rms_pct=float(100 * np.sqrt(np.mean(gate_half[1] ** 2))),
                               max_rel_th_change=float(np.max(np.abs(th_half / th - 1)))))
    print('gNa=%3.0f  V0=%.2f  rheobase=%.4f (50/100/200 ms: %s)  tau_rest=%.3f  '
          'tau_small_pulse=%.3f' % (g, out[g]['V0'], rheo, np.round(rheo_all, 4),
                                    tau_rest, tau_sp))
    for k, d in out[g]['fits'].items():
        print('     %-15s params %-30s RMS %.2f%%' % (k, np.round(d['p'], 3), d['rms_pct']))
    print('     EIF check: max |log(interp) - log(direct)| = %.2e' % eif_err)
    print('     dt/2: max threshold change %.2e; gate fit %s RMS %.2f%%'
          % (out[g]['dt_half']['max_rel_th_change'], np.round(gate_half[0], 3),
             out[g]['dt_half']['gate_rms_pct']))
    print('     latency HH  %s' % np.round(lat, 2))
    print('     latency t*+c %s' % np.round(lat_pred, 2))

# chronaxie: bisection on duration for I_th(d) = 2 b, all variants at once
rheos = np.array([out[g]['rheo'] for g in GNAS])
lo, hi = np.full(3, 0.1), np.full(3, 12.0)
for _ in range(25):
    mid = np.sqrt(lo * hi)
    above = hh_thresholds(mid, np.array(GNAS)) > 2 * rheos
    lo, hi = np.where(above, mid, lo), np.where(above, hi, mid)
for k, g in enumerate(GNAS):
    out[g]['chronaxie'] = float(np.sqrt(lo[k] * hi[k]))
    p = out[g]['fits']['Gate']['p']
    print('gNa=%3.0f  chronaxie=%.3f ms  (tau_rest ln2 = %.3f, leak RC = %.3f; gate fit '
          'tau ln(1+e^(c/tau)) = %.3f)' % (g, out[g]['chronaxie'], out[g]['tau_rest'] * np.log(2),
                                           rc_leak, p[1] * np.log(1 + np.exp(p[2] / p[1]))))

# ---------------------------------------------------------------- figure
fig, axs = plt.subplots(2, 2, figsize=(10.5, 7.4),
                        gridspec_kw=dict(hspace=0.42, wspace=0.28))
ax = axs[0, 0]
dd = np.geomspace(0.04, 60, 300)
for g in GNAS:
    o = out[g]
    b, tau, c = o['fits']['Gate']['p']
    ax.plot(DURS, np.array(o['th']) / o['rheo'], 'o', color=COLS[g], ms=4, label=LABELS[g])
    ax.plot(dd, b * gate_post(dd, tau, c) / o['rheo'], '-', color=COLS[g], lw=1.4)
ax.plot(dd, rc_curve(dd, tau_leak), '--', color=C_GRAY, lw=1.3,
        label=r'RC, leak $\tau_m$ = %.2f ms' % tau_leak)
ax.axhline(2.0, color='#999999', ls=':', lw=1.0)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(0.04, 60); ax.set_ylim(0.9, 60)
ax.set_xlabel('Pulse duration (ms)'); ax.set_ylabel('Threshold / rheobase')
ax.set_title('A. HH curves and gate fits (lines, Eq. 8b)', loc='left', fontsize=10.5,
             fontweight='bold')
ax.legend(fontsize=8, frameon=False, loc='upper right')

ax = axs[0, 1]
xs = np.arange(3)
vals = [out[g]['chronaxie'] for g in GNAS]
ax.bar(xs, vals, color=[COLS[g] for g in GNAS], width=0.6)
for x, g, v in zip(xs, GNAS, vals):
    ax.text(x, v + 0.05, '%.2f' % v, ha='center', fontsize=9.5, fontweight='bold')
    tr = out[g]['tau_rest'] * np.log(2)
    ax.plot([x - 0.36, x + 0.36], [tr, tr], color='#222222', lw=2.0)
    ax.text(x, 0.12, '$c$ = %.2f' % out[g]['fits']['Gate']['p'][2], ha='center',
            fontsize=8.2, color='white', fontweight='bold')
ax.axhline(rc_leak, color='#444444', ls='--', lw=1.3)
ax.text(2.45, rc_leak + 0.05, r'leak RC, $\tau_m\ln 2$ = %.2f ms' % rc_leak, ha='right',
        va='bottom', fontsize=8.2, color='#444444')
ax.text(-0.5, 2.0, r'black ticks: resting RC, $\tau_{\rm rest}\ln 2$', ha='left',
        va='bottom', fontsize=8.2)
ax.set_xticks(xs)
ax.set_xticklabels(['Low\n$g_{\\rm Na}$', 'Std\n$g_{\\rm Na}$', 'High\n$g_{\\rm Na}$'])
ax.set_ylim(0, 2.75); ax.set_xlim(-0.55, 2.55)
ax.set_ylabel('Chronaxie (ms)')
ax.set_title('B. Chronaxie vs two RC predictions', loc='left', fontsize=10.5,
             fontweight='bold')

ax = axs[1, 0]
o = out[120.0]
for k, lab, col, st in [('RC', 'RC (Lapicque)', C_GRAY, '--'), ('Weiss', 'Weiss', C_BLUE, ':'),
                        ('EIF', 'Exponential IF', C_GREEN, '-.'),
                        ('Gate_tau_rest', r'Gate, $\tau_m = \tau_{\rm rest}$', C_PLUM, '-'),
                        ('Gate', r'Gate, $\tau_m$ fitted', C_RED, '-')]:
    r = 100 * np.array(o['fits'][k]['resid'])
    ax.plot(DURS, r, st, marker='o', ms=3, color=col, lw=1.4,
            label='%s (RMS %.1f%%)' % (lab, o['fits'][k]['rms_pct']))
ax.axhline(0, color='#999999', lw=0.8)
ax.set_xscale('log'); ax.set_ylim(-25, 25)
ax.set_xlabel('Pulse duration (ms)'); ax.set_ylabel('Fit residual (% of threshold)')
ax.set_title(r'C. Residuals, std $g_{\rm Na}$', loc='left', fontsize=10.5, fontweight='bold')
ax.legend(fontsize=7.5, frameon=False, loc='lower right', ncol=1)

ax = axs[1, 1]
for g in GNAS:
    o = out[g]
    ax.plot(K_LAT, o['lat'], 'o', color=COLS[g], ms=5, label=LABELS[g] + ', HH')
    ax.plot(K_LAT, o['lat_pred'], '-', color=COLS[g], lw=1.4)
ax.plot([], [], '-', color='#444444', lw=1.4, label=r'gate prediction $t^* + c$')
ax.set_xscale('log')
ax.set_xticks([1.1, 1.5, 2, 3, 5, 8]); ax.set_xticklabels(['1.1', '1.5', '2', '3', '5', '8'])
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel('Step amplitude / rheobase'); ax.set_ylabel('First-spike latency (ms)')
ax.set_title('D. Held-out test: latency', loc='left', fontsize=10.5, fontweight='bold')
ax.legend(fontsize=7.8, frameon=False, loc='upper right')

fig.savefig('fig1_hh_chronaxie.pdf', bbox_inches='tight')
fig.savefig('fig1_hh_chronaxie.png', dpi=150, bbox_inches='tight')
json.dump({'durs': DURS.tolist(), 'tau_leak': tau_leak, 'rc_leak': rc_leak,
           'variants': {str(g): v for g, v in out.items()}},
          open('_out_fig1.json', 'w'), indent=1)
print('saved fig1_hh_chronaxie.pdf')
