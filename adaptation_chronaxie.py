"""Chronaxie after adaptation when the adaptation recovers during the test.

Section 5.1 and Prediction P1. A test pulse starts from rest (v = 0) at a
fixed delay after a conditioning step; the adaptation left by the step
decays with tau_w during the pulse and during any post-offset dwell.

  threshold : comparator V_theta + dV0 * exp(-t / tau_w)          (Eqs. 10-11)
  conductance: tau_m dv/dt = -v - G(t) (v - E_K) + R I,  G = G0 exp(-t / tau_w)
               (G dimensionless, g_K R), fixed comparator V_theta

tau_w = inf freezes the adaptation (quasi-static case). The gate is the
dwell timer of Eq. (7): a spike needs an uninterrupted suprathreshold
stretch of length c. Pulse-on scoring needs the stretch to be complete
before offset; post-offset scoring allows it up to 10 tau_m after offset.

Chronaxie is the minimum duration at twice a rheobase, which is defined
either asymptotically (infinitely long pulse) or operationally as the
threshold at the longest test duration T of the protocol.

Writes _out_adaptation_chronaxie.json. Runtime: about a minute.
"""
import json
import numpy as np

TAU_M = 15.0      # ms
R = 100.0         # MOhm
VTH = 15.0        # mV
C_GATE = 0.5      # ms
E_K = -15.0       # mV relative to rest
DT = 0.002        # ms
TAIL = 10 * TAU_M
B_REST = VTH / R  # nA


def trajectory(A, d, kind, size, tau_w):
    """Voltage and comparator on the grid t = k*DT, k = 1..n (exact for the
    linear ODE with the pulse ending at a grid point)."""
    n = int(round((d + TAIL) / DT))
    t = DT * np.arange(1, n + 1)
    if kind == 'conductance':
        G = size * np.exp(-t / tau_w) if np.isfinite(tau_w) else size * np.ones(n)
        Gm = size * np.exp(-(t - DT / 2) / tau_w) if np.isfinite(tau_w) else G
    else:
        Gm = np.zeros(n)
    # exact propagation step by step with coefficients at the step midpoint
    a = np.exp(-DT * (1.0 + Gm) / TAU_M)
    I = np.where(t - DT / 2 < d, A, 0.0)
    vinf = (R * I + Gm * E_K) / (1.0 + Gm)
    v = np.empty(n)
    x = 0.0
    # vectorised linear recursion x_{k} = a_k x_{k-1} + (1-a_k) vinf_k
    logP = np.cumsum(np.log(a))
    P = np.exp(logP)
    v = P * np.cumsum((1.0 - a) * vinf / P)
    if kind == 'threshold':
        th = VTH + (size * np.exp(-t / tau_w) if np.isfinite(tau_w) else size)
    else:
        th = VTH * np.ones(n)
    return t, v, th


def fires(A, d, kind, size, tau_w, window):
    t, v, th = trajectory(A, d, kind, size, tau_w)
    s = (v >= th).astype(np.int8)
    if window == 'on':
        s = s[: int(round(d / DT))]
    N = int(round(C_GATE / DT))
    if s.size < N:
        return False
    # longest run of consecutive suprathreshold samples
    edges = np.diff(np.concatenate(([0], s, [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return bool(np.any(ends - starts >= N))


def threshold_amp(d, kind, size, tau_w, window, rtol=1e-6):
    lo, hi = 0.0, 50.0 * B_REST
    while hi - lo > rtol * hi:
        mid = 0.5 * (lo + hi)
        if fires(mid, d, kind, size, tau_w, window):
            hi = mid
        else:
            lo = mid
    return hi


def min_duration(A, kind, size, tau_w, window, dmax=2000.0):
    lo, hi = 0, int(round(dmax / DT))
    if not fires(A, hi * DT, kind, size, tau_w, window):
        return float('nan')
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if fires(A, mid * DT, kind, size, tau_w, window):
            hi = mid
        else:
            lo = mid
    return hi * DT


def asymptotic_rheobase(kind, size, tau_w):
    if np.isfinite(tau_w):
        return B_REST                     # adaptation has recovered
    if kind == 'threshold':
        return (VTH + size) / R
    return (VTH * (1 + size) - size * E_K) / R


def main():
    out = {'params': dict(tau_m=TAU_M, R=R, V_theta=VTH, c=C_GATE, E_K=E_K,
                          dt=DT, tail=TAIL)}
    rows = []
    for window in ('post', 'on'):
        rows.append(dict(kind='rested', size=0.0, tau_w=None, window=window,
                         rheobase='asymptotic', b=B_REST,
                         chronaxie=min_duration(2 * B_REST, 'threshold', 0.0,
                                                np.inf, window)))
    cases = []
    for dV in (3.0, 7.5):
        cases.append(('threshold', dV, dV))
        cases.append(('conductance', dV / (VTH - E_K), dV))
    for kind, size, dV in cases:
        for tau_w in (np.inf, 100.0, 300.0):
            for window in ('post', 'on'):
                defs = [('asymptotic', asymptotic_rheobase(kind, size, tau_w))]
                if np.isfinite(tau_w):
                    for T in (100.0, 300.0):
                        defs.append(('T=%g ms' % T,
                                     threshold_amp(T, kind, size, tau_w, window)))
                for name, b in defs:
                    ch = min_duration(2 * b, kind, size, tau_w, window)
                    rows.append(dict(kind=kind, size=size, dV_equiv=dV,
                                     tau_w=None if not np.isfinite(tau_w) else tau_w,
                                     window=window, rheobase=name, b=b,
                                     chronaxie=ch))
                    print('%-11s dV=%4.1f tau_w=%5s %-4s %-11s b=%.4f nA  t_ch=%.3f ms'
                          % (kind, dV, tau_w, window, name, b, ch))
    for r in rows[:2]:
        print('rested %-4s t_ch=%.3f ms' % (r['window'], r['chronaxie']))
    out['rows'] = rows
    with open('_out_adaptation_chronaxie.json', 'w') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
