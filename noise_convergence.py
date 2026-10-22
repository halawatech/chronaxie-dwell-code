"""Time-step convergence at fixed noise intensity (Supplementary Methods, Table S6).

White current noise is specified by its per-step s.d. sigma at the reference
step dt0; this corresponds to a continuous-time white-noise intensity
sigma^2 * dt0 (nA^2 ms). Here each result is recomputed at dt0/2 and dt0/4 with
the per-step s.d. scaled as sigma * sqrt(dt0 / dt), so that the intensity, and
hence the stationary voltage variance (~ R^2 sigma^2 dt / (2 tau_m)), is fixed.

  A. Fig. 2:  d50 for c = 0.5 and 8 ms, both response windows (finer 0.02-ms
     duration grid, linear interpolation at P = 0.5).
  B. Fig. S7: mean latency and jitter (s.d. of first-spike time), with
     bootstrap 95% intervals for the jitter.
  C. Fig. S3: F1 at selected c and all three noise levels; plus sensitivity
     of F1 to the response tolerance after event offset (4, 8, 16 ms).
  D. Fig. S5: mean rate (spikes/s) at selected (c, f) probes.
"""
import json
import numpy as np
from msmodels import first_spike_batch, lif_gate

OUT = {}


def scaled(sigma, dt0, dt):
    return sigma * np.sqrt(dt0 / dt)


def d50_interp(durs, p):
    k = int(np.argmax(p >= 0.5))
    if k == 0:
        return float('nan')
    return float(durs[k - 1] + (0.5 - p[k - 1]) * (durs[k] - durs[k - 1]) / (p[k] - p[k - 1]))


# ---------------------------------------------------------------- A: Fig. 2
TAU, R, VTH, SIG0, DT0 = 2.0, 100.0, 15.0, 0.003, 0.02
A = 2 * VTH / R
OUT['fig2'] = {}
for dt in [DT0, DT0 / 2, DT0 / 4]:
    rng = np.random.default_rng(42)
    row = {}
    for window, tail in [('on', 0.0), ('post', 6 * TAU)]:
        for c in [0.5, 8.0]:
            centre = TAU * np.log(2) + c if window == 'on' else TAU * np.log(1 + np.exp(c / TAU))
            durs = np.round(np.arange(centre - 0.3, centre + 0.3 + 1e-9, 0.02), 4)
            p = np.array([np.mean(np.isfinite(first_spike_batch(
                A, d, tail, dt, TAU, R, VTH, c, scaled(SIG0, DT0, dt), 400, rng))) for d in durs])
            row['%s_c%g' % (window, c)] = (d50_interp(durs, p), float(centre))
    OUT['fig2']['%g' % dt] = row
    print('Fig.2  dt=%.4f  ' % dt + '  '.join('%s: d50 %.3f (eq %.3f)' % (k, *v)
                                              for k, v in row.items()), flush=True)

# ---------------------------------------------------------------- B: Fig. S7
TAU, VTH, AMP, DUR, DT0 = 5.0, 8.0, 0.16, 40.0, 0.02
OUT['figS7'] = {}
for dt in [DT0, DT0 / 2, DT0 / 4]:
    row = {}
    for sg in [0.005, 0.015, 0.05]:
        for c in [0.1, 1.0, 10.0]:
            rng = np.random.default_rng(7 + int(c * 10) + int(sg * 1e3))
            t = first_spike_batch(AMP, DUR, 0.0, dt, TAU, R, VTH, c,
                                  scaled(sg, DT0, dt), 600, rng)
            t = t[np.isfinite(t)]
            boot = [np.std(rng.choice(t, len(t))) for _ in range(500)]
            row['s%g_c%g' % (sg * 1000, c)] = (float(t.mean()), float(t.std()),
                                               float(np.quantile(boot, 0.025)),
                                               float(np.quantile(boot, 0.975)), int(len(t)))
    OUT['figS7']['%g' % dt] = row
    for sg in [5, 15, 50]:
        print('S7  dt=%.4f sigma=%2d pA  ' % (dt, sg) + '  '.join(
            'c=%g: lat %.3f jit %.4f [%.4f,%.4f]' % (c, *row['s%g_c%g' % (sg, c)][:4])
            for c in [0.1, 1.0, 10.0]), flush=True)

# ---------------------------------------------------------------- C: Fig. S3
TAU, VTH, T_REF, DT0 = 3.0, 8.0, 2.0, 0.05
SUST, TRANS, AMP_E = (25.0, 45.0), (1.0, 3.0), 0.16


def make_input(T, rng, sigma, dt, n_sust=8, n_trans=12):
    n = int(round(T / dt))
    I = np.zeros(n)
    events = []
    slots = np.linspace(60, T - 80, n_sust + n_trans)
    rng.shuffle(slots)
    for k, s in enumerate(slots):
        kind = 'sust' if k < n_sust else 'trans'
        d = rng.uniform(*(SUST if kind == 'sust' else TRANS))
        I[int(s / dt):int((s + d) / dt)] = AMP_E
        events.append((kind, s, s + d))
    return I + sigma * rng.standard_normal(n), events


def score(spikes, events, tol):
    n_s = sum(1 for e in events if e[0] == 'sust')
    det_s = det_t = 0
    inside = np.zeros(len(spikes), bool)
    for kind, a, b in events:
        hit = (spikes >= a) & (spikes <= b + tol)
        inside |= hit
        if hit.any():
            det_s += kind == 'sust'
            det_t += kind == 'trans'
    fp = det_t + int((~inside).sum())
    prec = det_s / max(1, det_s + fp)
    rec = det_s / max(1, n_s)
    return 2 * prec * rec / max(1e-9, prec + rec)


OUT['figS3'] = {}
for dt in [DT0, DT0 / 2]:
    row = {}
    for sg in [0.05, 0.15, 0.35]:
        for c in [1.0, 4.0, 12.0, 28.0]:
            f1 = {4.0: [], 8.0: [], 16.0: []}
            for rep in range(10):
                rr = np.random.default_rng(1000 + rep)
                I, ev = make_input(1500.0, rr, scaled(sg, DT0, dt), dt)
                sp, _ = lif_gate(I, dt, TAU, R, VTH, c, t_ref=T_REF)
                for tol in f1:
                    f1[tol].append(score(sp, ev, tol))
            row['s%g_c%g' % (sg, c)] = {('%g' % k): float(np.mean(v)) for k, v in f1.items()}
    OUT['figS3']['%g' % dt] = row
    for sg in [0.05, 0.15, 0.35]:
        print('S3  dt=%.3f sigma=%.2f  ' % (dt, sg) + '  '.join(
            'c=%g: F1 %s' % (c, '/'.join('%.2f' % row['s%g_c%g' % (sg, c)][t]
                                         for t in ('4', '8', '16')))
            for c in [1.0, 4.0, 12.0, 28.0]), flush=True)

# ---------------------------------------------------------------- D: Fig. S5
TAU, VTH, DT0, AMP_S, SIG_S = 3.0, 2.0, 0.05, 0.04, 0.004
T_S, BURN = 6500.0, 500.0
OUT['figS5'] = {}
for dt in [DT0, DT0 / 2]:
    row = {}
    n = int(round(T_S / dt)); t = np.arange(n) * dt
    for f in [2.0, 8.0, 20.0, 50.0]:
        rng = np.random.default_rng(42)
        I = AMP_S * np.sin(2 * np.pi * f * t * 1e-3) + scaled(SIG_S, DT0, dt) * rng.standard_normal(n)
        win = np.floor((T_S - BURN) * 1e-3 * f) / f * 1e3
        for c in [0.5, 8.0, 40.0]:
            sp, _ = lif_gate(I, dt, TAU, R, VTH, c, t_ref=2.0)
            row['c%g_f%g' % (c, f)] = float(np.sum((sp >= BURN) & (sp < BURN + win)) / (win * 1e-3))
    OUT['figS5']['%g' % dt] = row
    print('S5  dt=%.3f  ' % dt + '  '.join('%s %.1f' % kv for kv in row.items()), flush=True)

# ---------------------------------------------------------------- E: Fig. 6, small gates
# Same presynaptic spike trains as fig6_ctc.py (kappa = 20, seeds 0-2), generated
# on the 0.1-ms grid; for finer steps each presynaptic spike is placed at the same
# time and the alpha kernel is resampled at the finer step (unit area preserved).
NPRE, R0, MOD, TAU_SYN, QSPK = 200, 8.0, 0.9, 3.0, 4e-14
TAU_M6, VTH6, T6, BURN6, DT6 = 10.0, 8.0, 5000.0, 500.0, 0.1
n6 = int(T6 / DT6); t6 = np.arange(n6) * DT6


def presyn_counts(f, kappa, seed):
    rng = np.random.default_rng(1000 * seed + 42 + int(f) * 17 + kappa)
    phi = rng.vonmises(0.0, kappa, NPRE)
    counts = np.zeros(n6)
    for i in range(NPRE):
        lam = np.clip(R0 * (1 + MOD * np.cos(2 * np.pi * f * t6 * 1e-3 + phi[i])), 0, None)
        counts += (rng.random(n6) < lam * DT6 * 1e-3).astype(float)
    return counts


def epsc(counts, dt):
    m = int(round(DT6 / dt))
    fine = np.zeros(len(counts) * m)
    fine[::m] = counts
    kt = np.arange(0, 10 * TAU_SYN, dt)
    k = (kt / TAU_SYN) * np.exp(1 - kt / TAU_SYN)
    k = k / (k.sum() * dt * 1e-3)
    return np.convolve(fine, k, mode='full')[:len(fine)] * QSPK * 1e9


OUT['fig6'] = {}
for f in [40.0, 20.0, 6.0]:
    cnt = [presyn_counts(f, 20, s) for s in (0, 1, 2)]
    for dt in [0.1, 0.05, 0.025]:
        for c in [0.0, 0.5, 2.0]:
            r = []
            for k in range(3):
                sp, _ = lif_gate(epsc(cnt[k], dt), dt, TAU_M6, R, VTH6, c, t_ref=2.0)
                r.append(np.sum(sp >= BURN6) / ((T6 - BURN6) * 1e-3))
            OUT['fig6']['f%g_dt%g_c%g' % (f, dt, c)] = float(np.mean(r))
    print('Fig.6 f=%2.0f Hz  ' % f + '  '.join(
        'dt=%g: %s' % (dt, '/'.join('%.1f' % OUT['fig6']['f%g_dt%g_c%g' % (f, dt, c)]
                                    for c in (0.0, 0.5, 2.0))) for dt in (0.1, 0.05, 0.025)),
        flush=True)

# ---------------------------------------------------------------- F: Fig. 5, small gates
TAU_M5, TAU_R5, VTH5, DC5, AC5, T5, BURN5 = 10.0, 20.0, 8.0, 0.09, 0.05, 4000.0, 1000.0
FR5 = np.round(np.arange(4.0, 30.01, 0.5), 2)
OUT['fig5'] = {}
for dt in [0.05, 0.025]:
    n5 = int(T5 / dt); t5 = np.arange(n5) * dt
    for c in [0.0, 0.5, 3.0]:
        ok = []
        for f in FR5:
            sp, _ = lif_gate(DC5 + AC5 * np.sin(2 * np.pi * f * t5 * 1e-3), dt, TAU_M5, R,
                             VTH5, c, t_ref=2.0, resonator=dict(tau_r=TAU_R5))
            ncyc = int(np.floor((T5 - BURN5) * 1e-3 * f))
            win = ncyc / f * 1e3
            rate = np.sum((sp >= BURN5) & (sp < BURN5 + win)) / (win * 1e-3)
            ok.append(rate >= 0.5 * f)
        fs = FR5[np.array(ok)]
        OUT['fig5']['dt%g_c%g' % (dt, c)] = [float(fs.min()), float(fs.max())] if len(fs) else None
    print('Fig.5 dt=%.3f  passband c=0/0.5/3: %s' % (dt, '  '.join(
        str(OUT['fig5']['dt%g_c%g' % (dt, c)]) for c in (0.0, 0.5, 3.0))), flush=True)

json.dump(OUT, open('_out_noise_convergence.json', 'w'), indent=1)
print('saved _out_noise_convergence.json')
