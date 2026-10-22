"""Sweep behind the statement in Section 5.2 about gate + SFA and the predicted
peak at sqrt(f_SFA * f_gate) (Supplementary Methods, Table S7).

Same neuron, drive, noise and gain estimator as Fig. S4 (fig4_bandpass.py):
tau_m = 3 ms, V_theta = 8 mV, t_ref = 2 ms, DC = 0.09 nA, AC = 0.02 nA,
Ornstein-Uhlenbeck noise 0.006 nA (tau 5 ms), beta = 0.6 mV, gamma = 1; gain
from whole-cycle projections, complex coefficients averaged over realisations,
standard error from the realisations. The burn-in is max(0.8 s, 5 tau_w).

For each (c, tau_w):
  global : 12 log-spaced frequencies 0.5-40 Hz, 4 realisations each;
  local  : the predicted centre f_pred itself and its neighbours at half an
           octave and an octave (f_pred x 2^-1, 2^-1/2, 1, 2^1/2, 2), 8
           realisations each. A local peak at f_pred requires the gain there to
           exceed both half-octave neighbours by more than twice the combined
           standard error.
"""
import json
import numpy as np
from msmodels import lif_gate, ou_noise

TAU_M, R, VTH, T_REF, DT = 3.0, 100.0, 8.0, 2.0, 0.05
SIG_N, TAU_N, DC, AC, BETA, GAMMA = 0.006, 5.0, 0.09, 0.02, 0.6, 1.0
FR = np.round(np.geomspace(0.5, 40, 12), 3)
LOCAL = 2.0 ** np.array([-1.0, -0.5, 0.0, 0.5, 1.0])


def rate(c, adapt, burn, T_meas=6500.0, seed=3):
    n = int((T_meas + burn) / DT)
    rng = np.random.default_rng(seed)
    sp, _ = lif_gate(np.full(n, DC) + ou_noise(n, DT, SIG_N, TAU_N, rng), DT, TAU_M, R,
                     VTH, c, t_ref=T_REF, adapt=adapt)
    return np.sum(sp >= burn) / (T_meas * 1e-3)


def gain_at(c, f, adapt, burn, T_meas=12000.0, nreal=4):
    ncyc = max(1, int(np.floor(T_meas * 1e-3 * f)))
    T_win = ncyc / f * 1e3
    n = int(round((T_win + burn) / DT))
    t = np.arange(n) * DT
    drive = DC + AC * np.sin(2 * np.pi * f * t * 1e-3)
    zs = []
    for k in range(nreal):
        rng = np.random.default_rng(211 * k + int(f * 13) + 7)
        sp, _ = lif_gate(drive + ou_noise(n, DT, SIG_N, TAU_N, rng), DT, TAU_M, R, VTH, c,
                         t_ref=T_REF, adapt=adapt)
        sp = sp[(sp >= burn) & (sp < burn + T_win)] - burn
        zs.append(2 * np.sum(np.exp(1j * 2 * np.pi * f * sp * 1e-3)) / (T_win * 1e-3))
    zs = np.array(zs)
    zm = zs.mean()
    proj = np.real(zs * np.exp(-1j * np.angle(zm)))
    return float(abs(zm)), float(np.std(proj, ddof=1) / np.sqrt(nreal))


out = []
for c in [3.0, 10.0, 30.0, 60.0]:
    for tau_w in [100.0, 300.0, 1000.0]:
        ad = dict(tau_w=tau_w, beta=BETA, gamma=GAMMA)
        burn = max(800.0, 5 * tau_w)
        f_pred = np.sqrt(1000 / (2 * np.pi * c) * 1000 / (2 * np.pi * tau_w))
        g = np.array([gain_at(c, f, ad, burn) for f in FR])
        gl = np.array([gain_at(c, f_pred * m, ad, burn, nreal=8) for m in LOCAL])
        G, S = gl[:, 0], gl[:, 1]
        local_peak = bool(G[2] - G[1] > 2 * np.hypot(S[2], S[1]) and
                          G[2] - G[3] > 2 * np.hypot(S[2], S[3]))
        row = dict(c=c, tau_w=tau_w, burn_ms=burn, tonic=float(rate(c, ad, burn)),
                   f_pred=float(f_pred), f_peak=float(FR[int(np.argmax(g[:, 0]))]),
                   gains=g[:, 0].tolist(), gain_se=g[:, 1].tolist(),
                   local_freqs=(f_pred * LOCAL).tolist(), local_gains=G.tolist(),
                   local_se=S.tolist(), local_peak=local_peak,
                   local_rel=(G / G[2]).tolist())
        out.append(row)
        print('c=%4.0f tau_w=%5.0f tonic=%5.1f Hz  pred %.2f Hz  sampled peak %.2f Hz | '
              'local G(f/2..2f)/G(f_pred): %s  local peak: %s'
              % (c, tau_w, row['tonic'], f_pred, row['f_peak'],
                 ' '.join('%.2f' % x for x in row['local_rel']), local_peak), flush=True)
json.dump(dict(freqs=FR.tolist(), local_multipliers=LOCAL.tolist(), rows=out),
          open('_out_freqresp_sweep.json', 'w'), indent=1)
print('saved _out_freqresp_sweep.json')
