"""Manuscript Fig. 4 (main): the gate imposes a rate ceiling.
  A: f-I curves saturate at the gate-imposed ceiling 1/(c + t_ref).
  B: measured ceiling vs prediction.
Supplementary Fig. S4: modulation gain vs frequency for gate / SFA / gate+SFA
  (no bandpass at sqrt(f_SFA * f_gate)).
"""
from figstyle import *
from msmodels import lif_gate, ou_noise
import json

TAU_M, R, VTH, T_REF, DT = 3.0, 100.0, 8.0, 2.0, 0.05
SIG_N, TAU_N = 0.006, 5.0
CS = [2.0, 5.0, 10.0, 20.0]
COLS = {2.0: C_BLUE, 5.0: C_GREEN, 10.0: C_AMBER, 20.0: C_PLUM}


def rate(c, dc, adapt=None, T=8000.0, burn=1500.0, seed=3):
    n = int(T / DT)
    rng = np.random.default_rng(seed)
    I = np.full(n, dc) + ou_noise(n, DT, SIG_N, TAU_N, rng)
    sp, _ = lif_gate(I, DT, TAU_M, R, VTH, c, t_ref=T_REF, adapt=adapt)
    sp = sp[sp >= burn]
    return len(sp) / ((T - burn) * 1e-3)


def gain_at(c, dc, ac, f, adapt, T_meas=12000.0, burn=800.0, nreal=4):
    """Modulation of the output rate at the drive frequency f (spikes/s).

    For each realisation k the spike train after the burn-in is projected onto
    exp(i 2 pi f t) over an integer number of drive cycles (the longest that
    fits in T_meas), z_k = (2 / T) sum_spikes exp(i 2 pi f t_s); with whole
    cycles a constant rate contributes nothing. The complex coefficients are
    averaged across realisations (all share the same drive phase) before
    taking the modulus, which avoids the positive bias of averaging moduli.
    Returns (|<z>|, s.e., phase), the s.e. from the spread of the realisations'
    components along the mean direction.
    """
    ncyc = max(1, int(np.floor(T_meas * 1e-3 * f)))
    T_win = ncyc / f * 1e3
    n = int(round((T_win + burn) / DT))
    t = np.arange(n) * DT
    drive = dc + ac * np.sin(2 * np.pi * f * t * 1e-3)
    zs = []
    for k in range(nreal):
        rng = np.random.default_rng(211 * k + int(f * 13) + 7)
        I = drive + ou_noise(n, DT, SIG_N, TAU_N, rng)
        sp, _ = lif_gate(I, DT, TAU_M, R, VTH, c, t_ref=T_REF, adapt=adapt)
        sp = sp[(sp >= burn) & (sp < burn + T_win)] - burn
        zs.append(2 * np.sum(np.exp(1j * 2 * np.pi * f * sp * 1e-3)) / (T_win * 1e-3))
    zs = np.array(zs)
    zm = zs.mean()
    proj = np.real(zs * np.exp(-1j * np.angle(zm)))
    return float(abs(zm)), float(np.std(proj, ddof=1) / np.sqrt(nreal)), float(np.angle(zm))


# ---- A/B: rate ceiling -------------------------------------------------
DCS = np.round(np.geomspace(0.085, 1.2, 14), 4)
fi, ceil_meas = {}, {}
for c in CS:
    r = np.array([rate(c, d) for d in DCS])
    fi[c] = r
    ceil_meas[c] = r.max()
    print('c=%4.1f  ceiling meas=%6.1f Hz  pred=%6.1f Hz'
          % (c, r.max(), 1000.0 / (c + T_REF)))

# ---- C: modulation gain ------------------------------------------------
DC, AC = 0.09, 0.02
C_G, TAU_W, BETA, GAMMA = 10.0, 100.0, 0.6, 1.0
ad = dict(tau_w=TAU_W, beta=BETA, gamma=GAMMA)
f_gate = 1000 / (2 * np.pi * C_G)
f_sfa = 1000 / (2 * np.pi * TAU_W)
f_pred = np.sqrt(f_gate * f_sfa)
FR = np.round(np.geomspace(0.5, 40, 12), 3)
CONDS = {'Gate only': (C_G, None), 'SFA only': (0.0, ad), 'Gate + SFA': (C_G, ad)}
CC = {'Gate only': C_BLUE, 'SFA only': C_RED, 'Gate + SFA': C_GREEN}
gains, ses, phases, tonics = {}, {}, {}, {}
for nm, (c, a) in CONDS.items():
    tonics[nm] = rate(c, DC, adapt=a)
    g = np.array([gain_at(c, DC, AC, f, a) for f in FR])
    gains[nm], ses[nm], phases[nm] = g[:, 0], g[:, 1], g[:, 2]
    print('%-11s tonic=%.1f Hz  gain peak at %.1f Hz  gain: %s'
          % (nm, tonics[nm], FR[int(np.argmax(gains[nm]))],
             ' '.join('%.1f' % x for x in gains[nm])))

# amplitude check: halve the modulation (more realisations for the weaker signal)
AC2 = 0.01
gains2, ses2 = {}, {}
for nm in ('Gate only', 'Gate + SFA'):
    c, a = CONDS[nm]
    g = np.array([gain_at(c, DC, AC2, f, a, nreal=8) for f in FR])
    gains2[nm], ses2[nm] = g[:, 0], g[:, 1]
    print('%-11s AC=%.3f gain/AC: %s' % (nm, AC2, ' '.join('%.0f' % x for x in gains2[nm] / AC2)))
    print('%-11s AC=%.3f gain/AC: %s' % (nm, AC, ' '.join('%.0f' % x for x in gains[nm] / AC)))

fig = plt.figure(figsize=(8.6, 3.7))
gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 0.9], wspace=0.32)

ax = fig.add_subplot(gs[0])
for c in CS:
    ax.plot(DCS * R, fi[c], 'o-', color=COLS[c], ms=3.5, lw=1.7, label='$c$ = %g ms' % c)
    ax.axhline(1000.0 / (c + T_REF), color=COLS[c], ls=':', lw=1.2)
ax.set_xscale('log')
ax.set_xlabel('Drive $IR$ (mV)'); ax.set_ylabel('Firing rate (Hz)')
ax.set_title('A. Gate imposes a rate ceiling', loc='left', fontsize=10.5, fontweight='bold')
ax.legend(fontsize=8, frameon=False, loc='center right', bbox_to_anchor=(1.0, 0.76))
ax.text(0.97, 0.06, 'dotted: $1/(c+t_{\\rm ref})$', transform=ax.transAxes,
        ha='right', fontsize=8, color=C_GRAY, style='italic')

ax = fig.add_subplot(gs[1])
pred = [1000.0 / (c + T_REF) for c in CS]
meas = [ceil_meas[c] for c in CS]
lim = [0, max(pred) * 1.15]
ax.plot(lim, lim, '--', color=C_GRAY, lw=1.3, label='identity')
for c in CS:
    ax.plot(1000.0 / (c + T_REF), ceil_meas[c], 'o', color=COLS[c], ms=9)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel('Predicted $1/(c+t_{\\rm ref})$ (Hz)'); ax.set_ylabel('Measured max rate (Hz)')
ax.set_title('B. Ceiling matches prediction', loc='left', fontsize=10.5, fontweight='bold')
ax.legend(fontsize=8, frameon=False, loc='upper left')

fig.savefig('fig4_bandpass.pdf', bbox_inches='tight')
fig.savefig('fig4_bandpass.png', dpi=150, bbox_inches='tight')

fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.7), gridspec_kw=dict(wspace=0.34))
ax = axes[0]
for nm in CONDS:
    m = gains[nm].max()
    ax.errorbar(FR, gains[nm] / m, yerr=ses[nm] / m, fmt='o-', color=CC[nm], ms=3.5,
                lw=1.7, capsize=2, label=nm)
ax.axvline(f_pred, color=C_RED, ls='--', lw=1.3)
ax.text(f_pred * 1.08, 1.02, r'$\sqrt{f_{\rm SFA}f_{\rm gate}}$' + '\n= %.1f Hz' % f_pred,
        fontsize=8, color=C_RED)
ax.set_xscale('log'); ax.set_ylim(0, 1.18)
ax.set_xlabel('Modulation frequency (Hz)'); ax.set_ylabel('Gain / maximum gain')
ax.set_title('A. No bandpass at the predicted centre', loc='left',
             fontsize=10.5, fontweight='bold')
ax.legend(fontsize=8, frameon=False, loc='lower left')

ax = axes[1]
for nm in CONDS:
    ax.errorbar(FR, gains[nm] / AC, yerr=ses[nm] / AC, fmt='o-', color=CC[nm], ms=3.5,
                lw=1.7, capsize=2, label=nm)
ax.set_xscale('log'); ax.set_ylim(bottom=0)
ax.set_xlabel('Modulation frequency (Hz)'); ax.set_ylabel('Gain (spikes s$^{-1}$ nA$^{-1}$)')
ax.set_title('B. Absolute gain', loc='left', fontsize=10.5, fontweight='bold')

ax = axes[2]
for nm, mk in (('Gate only', 'o'), ('Gate + SFA', 's')):
    ax.errorbar(FR, gains[nm] / AC, yerr=ses[nm] / AC, fmt=mk + '-', color=CC[nm], ms=3.5,
                lw=1.5, capsize=2, label='%s, %.2f nA' % (nm, AC))
    ax.errorbar(FR, gains2[nm] / AC2, yerr=ses2[nm] / AC2, fmt=mk + '--', color=CC[nm],
                mfc='white', ms=3.5, lw=1.2, capsize=2, label='%s, %.2f nA' % (nm, AC2))
ax.set_xscale('log'); ax.set_ylim(bottom=0)
ax.set_xlabel('Modulation frequency (Hz)'); ax.set_ylabel('Gain (spikes s$^{-1}$ nA$^{-1}$)')
ax.set_title('C. Halving the modulation', loc='left', fontsize=10.5, fontweight='bold')
ax.legend(fontsize=7.2, frameon=False, loc='lower left')
fig.savefig('fig_freq_response.pdf', bbox_inches='tight')
fig.savefig('fig_freq_response.png', dpi=150, bbox_inches='tight')
json.dump({'cs': CS, 'ceil_meas': ceil_meas, 'dcs': list(map(float, DCS)),
           'fi': {str(c): list(map(float, fi[c])) for c in CS},
           'ceil_pred': {c: 1000.0 / (c + T_REF) for c in CS},
           'freqs': list(FR), 'ac': AC, 'gains': {k: list(v) for k, v in gains.items()},
           'gain_se': {k: list(v) for k, v in ses.items()},
           'phase': {k: list(v) for k, v in phases.items()},
           'ac_half': AC2, 'gains_half': {k: list(v) for k, v in gains2.items()},
           'gain_se_half': {k: list(v) for k, v in ses2.items()},
           'tonics': tonics, 'f_pred': f_pred}, open('_out_fig4.json', 'w'), indent=1)
print('saved fig4_bandpass.pdf, fig_freq_response.pdf')
