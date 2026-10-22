"""Manuscript Fig. 5: the resonance gatekeeper.

Neurons with identical subthreshold resonance, differing only in the gate
duration c. Rates are steady-state: the first BURN ms are discarded and spikes
are counted over an integer number of input cycles.
"""
from figstyle import *
from msmodels import lif_gate, resonator_impedance
import json

TAU_M, TAU_R, R, VTH, T_REF, DT = 10.0, 20.0, 100.0, 8.0, 2.0, 0.05
DC, AC = 0.09, 0.05
T, BURN = 4000.0, 1000.0
res = dict(tau_r=TAU_R)

ff = np.linspace(0.5, 40, 800)
Z = resonator_impedance(ff, TAU_M, TAU_R, R)
f_res = ff[int(np.argmax(Z))]
print('f_res = %.2f Hz   |Z|max = %.1f MOhm   |Z(0)| = %.1f MOhm'
      % (f_res, Z.max(), resonator_impedance(np.array([1e-6]), TAU_M, TAU_R, R)[0]))
print('v_dc = %.2f mV ; predicted peak at f_res = %.2f mV'
      % (DC * R / 2, DC * R / 2 + AC * Z.max()))

n = int(T / DT)
t = np.arange(n) * DT
FR = np.round(np.arange(4.0, 30.01, 0.5), 2)
CSWEEP = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]
CS_SHORT, CS_LONG = 3.0, 20.0


def run(c, f):
    I = DC + AC * np.sin(2 * np.pi * f * t * 1e-3)
    return lif_gate(I, DT, TAU_M, R, VTH, c, t_ref=T_REF, resonator=res)


def steady_rate(sp, f):
    ncyc = int(np.floor((T - BURN) * 1e-3 * f))
    win = ncyc / f * 1e3                      # ms, integer number of cycles
    k = np.sum((sp >= BURN) & (sp < BURN + win))
    return k / (win * 1e-3)


def band(fs, ok):
    fs = np.asarray(fs)[np.asarray(ok)]
    if len(fs) == 0:
        return None
    return float(fs.min()), float(fs.max())


rates, summary = {}, {}
for c in CSWEEP:
    rates[c] = np.array([steady_rate(run(c, f)[0], f) for f in FR])
    resp = band(FR, rates[c] >= 0.5 * FR)                  # >= 1 spike per 2 cycles
    lock = band(FR, np.abs(rates[c] - FR) <= 0.05 * FR)    # 1:1 locking
    summary[c] = dict(resp=resp, lock=lock)
    fmt = lambda b: ('%5.1f-%5.1f Hz (width %4.1f, centre %5.2f)'
                     % (b[0], b[1], b[1] - b[0], 0.5 * (b[0] + b[1])) if b else 'silent')
    print('c=%5.1f  rate>=f/2: %s   1:1 lock: %s' % (c, fmt(resp), fmt(lock)))

# traces at the grid frequency nearest the impedance peak
f_show = float(FR[int(np.argmin(np.abs(FR - f_res)))])
sp_s, v_s = run(CS_SHORT, f_show)
sp_l, v_l = run(CS_LONG, f_show)
_, v_pass = run(1e9, f_show)                   # gate never satisfied: passive voltage
v_late = v_pass[int(BURN / DT):]
dwell = (v_late >= VTH).sum() * DT / ((T - BURN) * 1e-3 * f_show)
print('f_show=%.1f Hz: c=%g -> %.1f Hz, c=%g -> %.1f Hz; passive dwell %.2f ms/cycle, '
      'peak %.2f mV' % (f_show, CS_SHORT, steady_rate(sp_s, f_show), CS_LONG,
                        steady_rate(sp_l, f_show), dwell, v_late.max()))

fig = plt.figure(figsize=(13, 7.2))
gs = fig.add_gridspec(2, 6, hspace=0.45, wspace=1.2)

ax = fig.add_subplot(gs[0, 0:2])
ax.plot(ff, Z / Z.max(), color='#222222', lw=2)
ax.axvline(f_res, color=C_GRAY, ls=':', lw=1.2)
ax.text(f_res + 0.6, 0.55, '$f_{\\rm res}$ = %.1f Hz' % f_res, fontsize=9, color=C_GRAY)
ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('$|Z(f)|$ (norm.)')
ax.set_title('A. Subthreshold impedance\n(identical for all $c$)', loc='left',
             fontsize=10.5, fontweight='bold')

ax = fig.add_subplot(gs[0, 2:4])
M = np.array([rates[c] / FR for c in CSWEEP])
df = FR[1] - FR[0]
im = ax.imshow(M, origin='lower', aspect='auto', cmap='Blues', vmin=0, vmax=1.0,
               extent=[FR[0] - df / 2, FR[-1] + df / 2, -0.5, len(CSWEEP) - 0.5])
ax.set_yticks(range(len(CSWEEP)))
ax.set_yticklabels(['%g' % c + (' (LIF)' if c == 0 else '') for c in CSWEEP], fontsize=8)
ax.axvline(f_res, color=C_RED, ls=':', lw=1.2)
ax.set_xlabel('Input frequency (Hz)'); ax.set_ylabel('Gate duration $c$ (ms)')
ax.set_title('B. Spikes per input cycle', loc='left', fontsize=10.5, fontweight='bold')
fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)

ax = fig.add_subplot(gs[0, 4:6])
w = np.array([(summary[c]['resp'][1] - summary[c]['resp'][0]) if summary[c]['resp'] else 0.0
              for c in CSWEEP])
silent = np.array([summary[c]['resp'] is None for c in CSWEEP])
ax.plot(np.array(CSWEEP)[~silent], w[~silent], 'o-', color=C_BLUE, lw=1.6, ms=5)
ax.plot(np.array(CSWEEP)[silent], w[silent], 'x', color=C_BLUE, ms=8, mew=1.8)
ax.set_xlabel('Gate duration $c$ (ms)'); ax.set_ylabel('Passband width (Hz)', color=C_BLUE)
ax.tick_params(axis='y', labelcolor=C_BLUE)
ax.set_xlim(-0.5, 21); ax.set_ylim(-0.8, 12.5)
axr = ax.twinx()
cs_ok = [c for c in CSWEEP if summary[c]['resp']]
axr.plot(cs_ok, [0.5 * sum(summary[c]['resp']) for c in cs_ok], 's--', color=C_RED,
         lw=1.4, ms=4, mfc='white')
axr.set_ylabel('Passband centre (Hz)', color=C_RED)
axr.tick_params(axis='y', labelcolor=C_RED)
axr.set_ylim(11.5, 15.5)
ax.axvline(dwell, color=C_GRAY, ls=':', lw=1.0)
ax.text(dwell + 0.4, 11.6, 'passive dwell\nat $f_{\\rm res}$ (%.1f ms)' % dwell,
        fontsize=7.5, color=C_GRAY, ha='left', va='top')
ax.text(20, 0.6, 'x = silent', fontsize=7.5, color=C_BLUE, ha='right')
ax.set_title('C. Passband vs $c$ (rate $\\geq f/2$)', loc='left',
             fontsize=10.5, fontweight='bold')

win = (BURN, BURN + 600.0)
for slot, (c, sp, v, col, lab) in zip([gs[1, 0:3], gs[1, 3:6]],
                                      [(CS_SHORT, sp_s, v_s, C_BLUE, 'D'),
                                       (CS_LONG, sp_l, v_l, C_RED, 'E')]):
    ax = fig.add_subplot(slot)
    m = (t >= win[0]) & (t < win[1])
    ax.plot(t[m], v[m], color=col, lw=0.8)
    ax.axhline(VTH, color='#444444', ls=':', lw=1.0)
    s_in = sp[(sp >= win[0]) & (sp < win[1])]
    for s in s_in:
        ax.plot([s, s], [VTH + 0.6, VTH + 2.2], color='#222222', lw=0.8)
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('$V$ (mV)')
    ax.set_title('%s. $c$ = %g ms at %.1f Hz: %.1f spikes/s (steady state)'
                 % (lab, c, f_show, steady_rate(sp, f_show)),
                 loc='left', fontsize=10.5, fontweight='bold')

fig.savefig('fig_gatekeeper.pdf', bbox_inches='tight')
fig.savefig('fig_gatekeeper.png', dpi=150, bbox_inches='tight')
TAB_F = [4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30]
json.dump({'f_res': float(f_res), 'freqs': list(map(float, FR)),
           'rates': {str(c): list(map(float, rates[c])) for c in CSWEEP},
           'summary': {str(c): summary[c] for c in CSWEEP},
           'table': {str(c): [float(rates[c][int(np.argmin(np.abs(FR - f)))]) for f in TAB_F]
                     for c in CSWEEP},
           'f_show': f_show, 'dwell_ms': float(dwell), 'peak_mV': float(v_late.max())},
          open('_out_fig5.json', 'w'), indent=1)
print('saved fig_gatekeeper.pdf')
