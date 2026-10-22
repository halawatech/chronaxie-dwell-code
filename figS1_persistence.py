"""Supplementary Figs. S2 & S3: persistence filtering demo, and precision/recall vs c and noise."""
from figstyle import *
from msmodels import lif_gate
import json

TAU_M, R, VTH, T_REF, DT = 3.0, 100.0, 8.0, 2.0, 0.05
SUST = (25.0, 45.0)      # sustained event duration range (ms)
TRANS = (1.0, 3.0)       # transient duration range (ms)
AMP = 0.16               # nA (2x rheobase)
TOL = 8.0                # ms after event offset still credited to the event


def make_input(T, rng, sigma, n_sust=5, n_trans=7):
    n = int(T / DT)
    I = np.zeros(n)
    events = []
    slots = np.linspace(60, T - 80, n_sust + n_trans)
    rng.shuffle(slots)
    for k, s in enumerate(slots):
        kind = 'sust' if k < n_sust else 'trans'
        d = rng.uniform(*(SUST if kind == 'sust' else TRANS))
        a, b = int(s / DT), int((s + d) / DT)
        I[a:b] = AMP
        events.append((kind, s, s + d))
    I = I + sigma * rng.standard_normal(n)
    return I, events


def score(spikes, events, tol=TOL):
    """Event-level scoring, as in the text: an event is 'detected' if it
    contains at least one spike. Transient events that are detected, and
    spikes falling outside any event, both count as false positives."""
    n_s = sum(1 for e in events if e[0] == 'sust')
    det_s = det_t = 0
    for kind, a, b in events:
        if np.any((spikes >= a) & (spikes <= b + tol)):
            if kind == 'sust':
                det_s += 1
            else:
                det_t += 1
    inside = np.zeros(len(spikes), bool)
    for _, a, b in events:
        inside |= (spikes >= a) & (spikes <= b + tol)
    n_noise = int((~inside).sum())
    prec = det_s / max(1, det_s + det_t + n_noise)
    rec = det_s / max(1, n_s)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return prec, rec, f1


# ---------------- Fig S1: demo ----------------
rng = np.random.default_rng(42)
T = 900.0
I, events = make_input(T, rng, 0.02)
t = np.arange(len(I)) * DT
sp_std, v_std = lif_gate(I, DT, TAU_M, R, VTH, 0.0, t_ref=T_REF)
sp_gate, v_gate = lif_gate(I, DT, TAU_M, R, VTH, 8.0, t_ref=T_REF)

fig, axes = plt.subplots(4, 1, figsize=(9.5, 7.6),
                         gridspec_kw=dict(height_ratios=[1, 1, 1, 0.8], hspace=0.55))
ax = axes[0]
ax.plot(t, I, color='#222222', lw=0.5)
for k, a, b in events:
    ax.axvspan(a, b, color=(C_GREEN if k == 'sust' else C_RED), alpha=0.25, lw=0)
ax.set_ylabel('Input (nA)')
ax.set_title('A. Mixed input (green = sustained %g–%g ms, red = transient %g–%g ms)'
             % (*SUST, *TRANS), loc='left', fontsize=10.5, fontweight='bold')

for ax, v, sp, name, col in [(axes[1], v_std, sp_std, 'Standard LIF', C_BLUE),
                             (axes[2], v_gate, sp_gate, 'Persistence LIF ($c$ = 8 ms)', C_PLUM)]:
    ax.plot(t, v, color=col, lw=0.6)
    ax.axhline(VTH, color='#444444', ls=':', lw=0.9)
    for s in sp:
        ax.plot([s, s], [VTH + 1, VTH + 5], color='#222222', lw=0.8)
    for k, a, b in events:
        ax.axvspan(a, b, color=(C_GREEN if k == 'sust' else C_RED), alpha=0.12, lw=0)
    ax.set_ylabel('$V$ (mV)')
    ax.set_title('%s. %s (%d spikes)' % ('B' if col == C_BLUE else 'C', name, len(sp)),
                 loc='left', fontsize=10.5, fontweight='bold')

det = {}
for name, sp in [('Std LIF', sp_std), ('Gate', sp_gate)]:
    for kind in ('sust', 'trans'):
        evs = [e for e in events if e[0] == kind]
        hit = sum(1 for (k, a, b) in evs if any(a <= s <= b + TOL for s in sp))
        det[(name, kind)] = hit / len(evs)
ax = axes[3]
keys = [('Std LIF', 'sust'), ('Gate', 'sust'), ('Std LIF', 'trans'), ('Gate', 'trans')]
ax.bar(range(4), [det[k] for k in keys],
       color=[C_BLUE, C_PLUM, C_BLUE, C_PLUM], width=0.6)
for i, k in enumerate(keys):
    ax.text(i, det[k] + 0.03, '%.2f' % det[k], ha='center', fontsize=9)
ax.set_xticks(range(4))
ax.set_xticklabels(['Sustained\n(Std LIF)', 'Sustained\n(Gate)',
                    'Transient\n(Std LIF)', 'Transient\n(Gate)'], fontsize=8.5)
ax.set_ylim(0, 1.18); ax.set_ylabel('Detection rate')
ax.set_title('D. Selectivity: sustained vs. transient', loc='left',
             fontsize=10.5, fontweight='bold')
fig.savefig('fig_persistence.pdf', bbox_inches='tight')
fig.savefig('fig_persistence.png', dpi=150, bbox_inches='tight')
print('S1 detection:', {'%s/%s' % k: round(v, 2) for k, v in det.items()})

# ---------------- Fig S2: precision / recall ----------------
CS = [0.1, 0.3, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 22.0, 28.0, 34.0, 40.0]
NREP = 10
SIGS = [0.05, 0.15, 0.35]
COLS = {0.05: C_PLUM, 0.15: C_RED, 0.35: C_AMBER}
P, Rc, F, FSE = {}, {}, {}, {}
for sg in SIGS:
    p_, r_, f_, se_ = [], [], [], []
    for c in CS:
        ps, rs, fs = [], [], []
        for rep in range(NREP):
            rr = np.random.default_rng(1000 + rep)
            I, ev = make_input(1500.0, rr, sg, n_sust=8, n_trans=12)
            sp, _ = lif_gate(I, DT, TAU_M, R, VTH, c, t_ref=T_REF)
            a, b, cc = score(sp, ev)
            ps.append(a); rs.append(b); fs.append(cc)
        p_.append(np.mean(ps)); r_.append(np.mean(rs)); f_.append(np.mean(fs))
        se_.append(np.std(fs, ddof=1) / np.sqrt(NREP))
    P[sg], Rc[sg], F[sg], FSE[sg] = map(np.array, (p_, r_, f_, se_))
    i = int(np.argmax(F[sg]))
    near = [CS[k] for k in range(len(CS)) if F[sg][k] >= F[sg][i] - FSE[sg][i]]
    print('sigma=%.2f nA  F1 max %.3f+-%.3f at c=%g ms; within 1 SE of max for c=%s'
          % (sg, F[sg][i], FSE[sg][i], CS[i], near))
    print('   precision: %s' % ' '.join('%.2f' % x for x in P[sg]))
    print('   recall   : %s' % ' '.join('%.2f' % x for x in Rc[sg]))
    print('   F1       : %s' % ' '.join('%.3f' % x for x in F[sg]))

fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
for ax, D, ttl, lab in [(axes[0], P, 'A. Precision', 'Precision'),
                        (axes[1], Rc, 'B. Recall', 'Recall'),
                        (axes[2], F, 'C. F1 score', 'F1')]:
    for sg in SIGS:
        ax.errorbar(CS, D[sg], yerr=(FSE[sg] if D is F else None), fmt='o-',
                    color=COLS[sg], lw=1.8, ms=4, capsize=2,
                    label=r'$\sigma$ = %.2f nA' % sg)
    ax.set_xlabel('Gate duration $c$ (ms)'); ax.set_ylabel(lab)
    ax.set_ylim(-0.05, 1.08)
    ax.set_title(ttl, loc='left', fontsize=10.5, fontweight='bold')
axes[0].legend(fontsize=8, frameon=False, loc='lower right')
plt.tight_layout()
fig.savefig('fig_precision_recall.pdf'); fig.savefig('fig_precision_recall.png', dpi=150)
json.dump({'cs': CS, 'F1': {str(k): list(v) for k, v in F.items()},
           'F1_se': {str(k): list(v) for k, v in FSE.items()},
           'prec': {str(k): list(v) for k, v in P.items()},
           'rec': {str(k): list(v) for k, v in Rc.items()}},
          open('_out_figS2.json', 'w'), indent=1)
print('saved fig_persistence.pdf, fig_precision_recall.pdf')
