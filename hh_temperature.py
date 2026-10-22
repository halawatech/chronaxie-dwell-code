"""Temperature scaling of HH chronaxie (Section 3.2, temperature note;
Supplementary Methods).

Standard HH (gNa = 120 mS/cm^2) with all gating rates multiplied by
phi = 3^((T - 6.3)/10); capacitance, conductances and reversal potentials are
unchanged. Steady-state gate values do not depend on phi, so the resting
membrane, and its chord time constant tau_rest, are the same at every T.
Spikes after pulse offset count (V reaches 0 mV within 30 ms of offset).

For each temperature: rheobase (60-ms pulse), chronaxie (bisection on
duration for threshold = 2 x rheobase), both at two time steps, and an
all-or-none check: the peak voltage reached at 0.98x and 1.02x the threshold
amplitude for a chronaxie-length pulse. A jump of tens of millivolts marks a
regenerative spike; similar peaks either side of 0 mV mark a graded response
(heat block), for which a 0-mV criterion does not define a threshold.
"""
import json
import numpy as np
from msmodels import hh_first_spike, hh_thresholds, hh_resting_tau

G = 120.0
out = dict(tau_rest=hh_resting_tau(G))
for T in [6.3, 15.0, 20.0, 25.0, 30.0, 37.0]:
    phi = 3.0 ** ((T - 6.3) / 10.0)
    rows = []
    for dt in [min(0.01, 0.1 / phi), min(0.005, 0.05 / phi)]:
        rheo = float(hh_thresholds(np.array([60.0]), G, dt=dt, phi=phi, rtol=1e-4)[0])
        lo, hi = 0.01, 10.0
        for _ in range(18):
            mid = np.sqrt(lo * hi)
            th = hh_thresholds(np.array([mid]), G, dt=dt, phi=phi, rtol=1e-4)[0]
            lo, hi = (lo, mid) if th <= 2 * rheo else (mid, hi)
        chron = float(np.sqrt(lo * hi))
        thr = hh_thresholds(np.array([chron]), G, dt=dt, phi=phi, rtol=1e-5)[0]
        _, rec = hh_first_spike(np.array([0.98, 1.02]) * thr, chron, G, dt=dt, tail=20.0,
                                phi=phi, record=True)
        peaks = [float(rec[:, 0].max()), float(rec[:, 1].max())]
        rows.append(dict(dt=dt, rheobase=rheo, chronaxie=chron, peak_098=peaks[0],
                         peak_102=peaks[1], all_or_none=bool(peaks[1] - peaks[0] > 30.0)))
        print('T=%4.1f C  phi=%5.2f  dt=%.4f  rheobase=%8.3f  chronaxie=%.3f ms  '
              'peak V at 0.98x / 1.02x threshold: %6.1f / %6.1f mV  %s'
              % (T, phi, dt, rheo, chron, *peaks,
                 'spike' if rows[-1]['all_or_none'] else 'GRADED (heat block)'), flush=True)
    out[str(T)] = dict(phi=phi, runs=rows)
print('resting chord time constant tau_rest = %.3f ms at every temperature '
      '(tau_rest ln 2 = %.3f ms)' % (out['tau_rest'], out['tau_rest'] * np.log(2)))

# ---------------------------------------------------------------------------
# Does the reduction with tau_m fixed at tau_rest survive warming? Full
# strength-duration curves (22 durations) at the temperatures that give
# all-or-none spikes, fitted with Eq. (8b):
#   free     : b, tau_m, c >= 0 free (phenomenological description)
#   rest     : tau_m = tau_rest, b and c >= 0 free
#   rest_neg : tau_m = tau_rest, b free, c unconstrained in sign
#   rest_b   : tau_m = tau_rest, b fixed at the measured rheobase, c unconstrained
#   rc       : Eq. (3), b and tau_m free
# plus the dwell implied by chronaxie alone, c = tau_rest ln(expm1(t_ch / tau_rest)).
# ---------------------------------------------------------------------------
from scipy.optimize import least_squares

DURS = np.geomspace(0.05, 50.0, 22)
tr = out['tau_rest']
gate = lambda d, tau, c: 1 + np.exp(c / tau) / np.expm1(d / tau)
rcf = lambda d, tau: 1 / (-np.expm1(-d / tau))


def fit(model, y, p0, lb, ub):
    r = least_squares(lambda p: np.log(p[0]) + np.log(model(DURS, *p[1:])) - np.log(y),
                      p0, bounds=(lb, ub))
    return r.x, float(100 * np.sqrt(np.mean(r.fun ** 2)))


out['reduction'] = {}
for T in [6.3, 15.0, 20.0]:
    phi = 3.0 ** ((T - 6.3) / 10.0)
    dt = min(0.01, 0.1 / phi)
    th = hh_thresholds(DURS, G, dt=dt, phi=phi)
    rheo = float(hh_thresholds(np.array([60.0]), G, dt=dt, phi=phi)[0])
    t_ch = out[str(T)]['runs'][0]['chronaxie']
    res = {
        'free': fit(gate, th, [rheo, 1.5, 0.3], [0, 0.02, 0.0], [1e3, 50, 20]),
        'rest': fit(lambda d, c: gate(d, tr, c), th, [rheo, 0.3], [0, 0.0], [1e3, 20]),
        'rest_neg': fit(lambda d, c: gate(d, tr, c), th, [rheo, 0.0], [0, -10.0], [1e3, 20]),
        'rc': fit(rcf, th, [rheo, 1.5], [0, 0.02], [1e3, 50]),
    }
    rb = least_squares(lambda p: np.log(rheo) + np.log(gate(DURS, tr, p[0])) - np.log(th),
                       [0.0], bounds=([-10.0], [20.0]))
    res['rest_b'] = (np.array([rheo, rb.x[0]]), float(100 * np.sqrt(np.mean(rb.fun ** 2))))
    c_implied = float(tr * np.log(np.expm1(t_ch / tr)))
    out['reduction'][str(T)] = dict(
        dt=dt, rheobase=rheo, chronaxie=t_ch, c_implied_from_chronaxie=c_implied,
        th=th.tolist(), fits={k: dict(p=list(map(float, p)), rms_pct=rms,
                                      b_over_rheobase=float(p[0] / rheo))
                              for k, (p, rms) in res.items()})
    print('T=%4.1f C  chronaxie %.3f ms -> implied c (tau_rest fixed) %+.3f ms' % (T, t_ch, c_implied))
    for k, (p, rms) in res.items():
        print('     %-8s params %-32s RMS %5.2f%%  b/rheobase %.3f'
              % (k, np.round(p, 3), rms, p[0] / rheo))
json.dump(out, open('_out_hh_temperature.json', 'w'), indent=1)
