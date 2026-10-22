"""
Shared models for the chronaxie manuscript figures.

Units: time ms, voltage mV, current nA, resistance MOhm  (nA * MOhm = mV).
HH block uses its own units: uA/cm^2, mS/cm^2, uF/cm^2, mV, ms.
"""
import numpy as np

# ----------------------------------------------------------------------
# Hodgkin-Huxley (standard squid parameterisation, T = 6.3 C)
# ----------------------------------------------------------------------
HH = dict(C=1.0, gL=0.3, EL=-54.4, gK=36.0, EK=-77.0, ENa=50.0)
V_REST = -65.0


def _rates(V):
    # guard the removable singularities (np.where evaluates both branches,
    # so silence the 0/0 warning from the branch that is discarded)
    with np.errstate(invalid='ignore', divide='ignore'):
        a_m = np.where(np.abs(V + 40) < 1e-7, 1.0,
                       0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10)))
        a_n = np.where(np.abs(V + 55) < 1e-7, 0.1,
                       0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10)))
    b_m = 4.0 * np.exp(-(V + 65) / 18)
    a_h = 0.07 * np.exp(-(V + 65) / 20)
    b_h = 1.0 / (1 + np.exp(-(V + 35) / 10))
    b_n = 0.125 * np.exp(-(V + 65) / 80)
    return a_m, b_m, a_h, b_h, a_n, b_n


def _gates_inf(V):
    a_m, b_m, a_h, b_h, a_n, b_n = _rates(V)
    return a_m / (a_m + b_m), a_h / (a_h + b_h), a_n / (a_n + b_n)


def hh_rest_state(gNa=120.0):
    """Resting state for the given gNa: the voltage at which the steady-state
    ionic current is zero, with gates at their steady-state values. Each gNa
    variant therefore starts at its own equilibrium, with no initial drift."""
    from scipy.optimize import brentq
    p = HH

    def i_ion(V):
        m, h, n = _gates_inf(V)
        return (gNa * m ** 3 * h * (V - p['ENa']) + p['gK'] * n ** 4 * (V - p['EK'])
                + p['gL'] * (V - p['EL']))
    V0 = brentq(i_ion, -80.0, -55.0)
    m, h, n = _gates_inf(V0)
    return V0, m, h, n


def hh_resting_tau(gNa=120.0):
    """Chord time constant of the resting membrane, C / g_rest, with every
    conductance (leak, K, Na) evaluated at its resting steady state."""
    V0, m, h, n = hh_rest_state(gNa)
    g_rest = HH['gL'] + HH['gK'] * n ** 4 + gNa * m ** 3 * h
    return HH['C'] / g_rest


def hh_first_spike(amps, durs, gNa, dt=0.01, tail=30.0, phi=1.0, record=False,
                   v_spike=0.0):
    """Rectangular pulses, vectorised over elements: element k receives
    amps[k] (uA/cm^2) for durs[k] ms, starting from the resting equilibrium of
    its own gNa (scalar or per element). Gating rates are multiplied by phi
    (temperature scaling). Forward Euler, gates updated before V. The pulse
    delivers exactly amps[k] * durs[k] of charge: the step containing the
    offset carries the fraction of the step that lies inside the pulse.

    Returns the first time V reaches v_spike within durs[k] + tail ms (NaN if
    none); with record=True also the voltage traces (time x elements).
    """
    amps = np.atleast_1d(np.asarray(amps, float))
    durs = np.broadcast_to(np.asarray(durs, float), amps.shape)
    gNa = np.broadcast_to(np.asarray(gNa, float), amps.shape)
    rest = {g: hh_rest_state(g) for g in np.unique(gNa)}
    V, m, h, n = (np.array([rest[g][k] for g in gNa]) for k in range(4))
    p = HH
    tend = durs + tail
    first = np.full(amps.shape, np.nan)
    rec = []
    for i in range(int(round(tend.max() / dt))):
        t = i * dt
        # pulse ends exactly at durs: the step containing the offset carries
        # the fraction of the step that lies inside the pulse
        I = amps * np.clip((durs - t) / dt, 0.0, 1.0)
        a_m, b_m, a_h, b_h, a_n, b_n = _rates(V)
        m = m + dt * phi * (a_m * (1 - m) - b_m * m)
        h = h + dt * phi * (a_h * (1 - h) - b_h * h)
        n = n + dt * phi * (a_n * (1 - n) - b_n * n)
        V = V + dt * (I - gNa * m ** 3 * h * (V - p['ENa']) - p['gK'] * n ** 4 * (V - p['EK'])
                      - p['gL'] * (V - p['EL'])) / p['C']
        new = np.isnan(first) & (V >= v_spike) & (t + dt <= tend + 1e-9)
        first[new] = t + dt
        if record:
            rec.append(V.copy())
        elif not np.isnan(first).any():
            break
    return (first, np.array(rec)) if record else first


def hh_thresholds(durs, gNa, lo=0.0, hi=2000.0, rtol=2e-4, dt=0.01, phi=1.0, tail=30.0):
    """Threshold amplitude for each pulse duration (bisection, vectorised over
    durations; gNa scalar or per duration). A response counts as a spike if V
    reaches 0 mV within the pulse or the following `tail` ms."""
    durs = np.atleast_1d(np.asarray(durs, float))
    lo = np.full(durs.shape, lo)
    hi = np.full(durs.shape, hi)
    while np.max((hi - lo) / hi) > rtol:
        mid = 0.5 * (lo + hi)
        fired = np.isfinite(hh_first_spike(mid, durs, gNa, dt=dt, tail=tail, phi=phi))
        hi = np.where(fired, mid, hi)
        lo = np.where(fired, lo, mid)
    return hi


# ----------------------------------------------------------------------
# Persistence-gated leaky integrate-and-fire
# ----------------------------------------------------------------------
def lif_gate(I, dt, tau_m, R, Vth, c, t_ref=2.0, adapt=None, resonator=None):
    """Simulate a persistence-gated LIF.

    I         : current trace (nA)
    c         : persistence requirement (ms); c<=0 -> standard LIF
    adapt     : None or dict(tau_w=, beta=, gamma=)  threshold-elevation SFA
    resonator : None or dict(tau_r=)  adds resonance variable u (Eqs. S4-S5)

    Timing rule. Step i advances v from time i*dt to (i+1)*dt (forward Euler)
    and then compares it with threshold; v_tr[i] is v at (i+1)*dt. A spike is
    emitted at the end of the step in which the count of consecutive
    suprathreshold samples reaches N = round(c/dt) (at least 1) and is stamped
    (i+1)*dt. Because the true crossing lies somewhere in the step before the
    first suprathreshold sample, the realised dwell after the crossing lies
    between (N-1)*dt and N*dt, i.e. between c - dt and c.

    Returns (spike_times_ms, v_trace_mV)
    """
    n = len(I)
    v = 0.0
    u = 0.0
    w = 0.0
    run = 0                      # consecutive suprathreshold samples
    ref = 0
    need = max(1, int(round(c / dt))) if c > 0 else 1
    v_tr = np.empty(n)
    spikes = []
    n_ref = int(round(t_ref / dt))
    for i in range(n):
        if resonator is None:
            dv = (-v + R * I[i]) / tau_m
        else:
            dv = (-v - u + R * I[i]) / tau_m
            u += dt * (v - u) / resonator['tau_r']
        v += dt * dv
        if adapt is not None:
            w += dt * (-w / adapt['tau_w'])
            th = Vth + adapt['gamma'] * w
        else:
            th = Vth
        v_tr[i] = v
        if ref > 0:
            ref -= 1
            run = 0
            continue
        if v >= th:
            run += 1
            if run >= need:
                spikes.append((i + 1) * dt)
                v = 0.0
                run = 0
                ref = n_ref
                if adapt is not None:
                    w += adapt['beta']
        else:
            run = 0
    return np.asarray(spikes), v_tr


def ou_noise(n, dt, sigma, tau, rng):
    """Ornstein-Uhlenbeck current noise (nA)."""
    x = np.empty(n)
    x[0] = 0.0
    a = np.exp(-dt / tau)
    s = sigma * np.sqrt(1 - a ** 2)
    g = rng.standard_normal(n)
    for i in range(1, n):
        x[i] = a * x[i - 1] + s * g[i]
    return x


def resonator_impedance(f_hz, tau_m, tau_r, R):
    """|Z(f)| for Eqs. S4-S5 (Eq. S6; tau in ms, f in Hz)."""
    w = 2 * np.pi * f_hz * 1e-3          # rad/ms
    num = R * np.sqrt(1 + (w * tau_r) ** 2)
    den = np.sqrt((2 - w ** 2 * tau_m * tau_r) ** 2 + (w * (tau_m + tau_r)) ** 2)
    return num / den


def logistic(x, x0, k):
    z = np.clip(-k * (x - x0), -60, 60)
    return 1.0 / (1.0 + np.exp(z))


def d50_from_curve(durations, p, k0=5.0):
    """Half-maximum duration from a logistic fit. If the fit fails, falls back
    to linear interpolation at the first upward crossing of 0.5 and warns."""
    from scipy.optimize import curve_fit
    durations, p = np.asarray(durations, float), np.asarray(p, float)
    try:
        popt, _ = curve_fit(logistic, durations, p, p0=[np.median(durations), k0],
                            maxfev=20000)
        return popt[0]
    except Exception:
        import warnings
        warnings.warn('logistic fit failed; using linear interpolation for d50')
        idx = int(np.argmax(p >= 0.5))
        if idx == 0:
            return np.nan
        x0, x1, y0, y1 = durations[idx - 1], durations[idx], p[idx - 1], p[idx]
        return x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0)


def d50_bootstrap(durations, p, n_trial, n_boot=200, rng=None):
    """d50 with a binomial parametric-bootstrap standard error."""
    rng = np.random.default_rng(0) if rng is None else rng
    est = d50_from_curve(durations, p)
    boots = [d50_from_curve(durations, rng.binomial(n_trial, p) / n_trial)
             for _ in range(n_boot)]
    return est, float(np.nanstd(boots))


def first_spike_batch(A, d_on, t_tail, dt, tau_m, R, Vth, c, sigma, n_trial, rng):
    """First-spike time of a persistence-gated LIF for many independent trials
    at once (vectorised over trials).

    Rectangular pulse of amplitude A (nA) lasting d_on ms, followed by t_tail ms
    of zero mean drive; white current noise of s.d. sigma (nA) per time step
    throughout, as in lif_gate. Same update order, gate rule and timing rule
    as lif_gate (spike stamped at the end of the step that completes the count).
    t_tail = 0 scores spikes during the pulse only (the pulse-on window of
    Eq. 8); t_tail > 0 also counts spikes after pulse offset (Eq. 8b).

    Returns an array of first-spike times (ms); NaN where no spike occurred.
    """
    n_on = max(1, int(round(d_on / dt)))
    n = n_on + int(round(t_tail / dt))
    need = max(1, int(round(c / dt))) if c > 0 else 1
    v = np.zeros(n_trial)
    run = np.zeros(n_trial, int)
    first = np.full(n_trial, np.nan)
    live = np.ones(n_trial, bool)
    for i in range(n):
        drive = A if i < n_on else 0.0
        I = drive + sigma * rng.standard_normal(n_trial)
        v += dt * (-v + R * I) / tau_m
        above = v >= Vth
        run = np.where(above, run + 1, 0)
        hit = live & (run >= need)
        if hit.any():
            first[hit] = (i + 1) * dt
            live &= ~hit
            if not live.any():
                break
        # once the pulse is off and the voltage has fallen back below
        # threshold, a noise-free membrane cannot re-cross; stop early
        if i >= n_on and not (live & above).any() and (v[live] < 0.9 * Vth).all():
            break
    return first


def rate_gain(spikes, T, f_hz, dt_bin=1.0):
    """Amplitude of the rate modulation at f_hz (spikes/s), via projection
    onto sin/cos at that frequency."""
    if len(spikes) < 4:
        return 0.0
    edges = np.arange(0, T + dt_bin, dt_bin)
    r, _ = np.histogram(spikes, bins=edges)
    r = r / (dt_bin * 1e-3)               # -> Hz
    t = (edges[:-1] + dt_bin / 2) * 1e-3  # s
    w = 2 * np.pi * f_hz
    s = np.sin(w * t)
    co = np.cos(w * t)
    a = 2 * np.mean(r * s)
    b = 2 * np.mean(r * co)
    return np.hypot(a, b)
