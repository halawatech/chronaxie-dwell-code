# Spike-Initiation Dwell Time — simulation code

Code and data for *How Much of Chronaxie Is Spike-Initiation Dwell? Closed
Forms, a Hodgkin–Huxley Test and the Limits of Strength–Duration
Measurements*.

## Environment

- Python 3.12.10, NumPy 2.4.6, SciPy 1.18.0, Matplotlib 3.10.9 (the versions
  used to generate the figures; other recent versions should work).

## Installation and use

```bash
pip install -r requirements.txt
python fig3_duration.py
```

Run each script from this folder. It writes its figure (PDF and PNG) and,
where applicable, a JSON file with the numbers quoted in the paper. The
`_out_*.json` files in this repository are the outputs used for the
manuscript.

## Layout

| File | Purpose |
|---|---|
| `msmodels.py` | Shared models: Hodgkin–Huxley (per-gNa resting equilibrium, resting chord time constant, vectorised first-spike and threshold search), persistence-gated LIF (`lif_gate`), vectorised first-spike simulator (`first_spike_batch`), resonator impedance, OU noise, d50 estimation |
| `figstyle.py` | Shared Matplotlib style and colours, imported by every figure script. It sets appearance only (fonts, colours, line widths) and re-exports NumPy and Matplotlib; it computes nothing. It was rewritten because the original file was lost, and every figure was generated with this version. |

## Script → figure map

Script names come from an earlier numbering and do **not** match the figure
numbers of the paper.

| Script | Item in the paper | Output | Approx. runtime |
|---|---|---|---|
| `fig3_duration.py` | Fig. 1 (the response window: pulse-on vs post-offset scoring) | `fig3_duration_selectivity.pdf`, `_out_fig3.json` | ~2 min |
| `fig1_hh.py` | Fig. 2 and Section 3.2 (HH strength–duration curves; gate, RC, Weiss and EIF fits with a direct-inversion check of the EIF; half-step check; latency test) | `fig1_hh_chronaxie.pdf`, `_out_fig1.json` | ~20 min |
| `fig_identifiability.py` | Fig. 3 (parameter recovery: can c be identified?) | `fig_identifiability.pdf`, `_out_identifiability.json` | ~1 h; `--plot-only` redraws from the JSON |
| `figS5_realistic_taum.py` | Fig. S1 (the response window at tau_m = 2, 8, 15 and 30 ms) | `fig_realistic_taum.pdf`, `_out_figS5.json` | ~5 min |
| `figS4_precision.py` | Fig. S2 (latency vs jitter) | `fig_spike_precision.pdf`, `_out_figS4.json` | ~1 min |
| `hh_temperature.py` | Section 3.2 temperature paragraph, Table S1 and Supplementary Methods (HH chronaxie at 6.3–37 °C, two time steps, all-or-none check; full-curve fits of the reduction at 6.3, 15 and 20 °C) | `_out_hh_temperature.json` | ~45 min |
| `noise_convergence.py` | Table S2 (time-step convergence at fixed noise intensity for Figs. 1 and S2; jitter bootstrap intervals) | `_out_noise_convergence.json` | ~10 min |

Runtimes are approximate, measured on an Intel Core i7-13700F (16 cores, 24
threads) with 32 GB RAM, one Python process per script.

### Other scripts

These belong to analyses from an earlier, longer version of the manuscript
and are not reported in the current paper. They are kept so that the
archived version stays complete.

| Script | Analysis | Output |
|---|---|---|
| `fig4_bandpass.py` | Rate ceiling and frequency response | `fig4_bandpass.pdf`, `fig_freq_response.pdf`, `_out_fig4.json` |
| `fig5_gatekeeper.py` | Resonance gatekeeper | `fig_gatekeeper.pdf`, `_out_fig5.json` |
| `fig6_ctc.py` | Feedforward recruitment | `fig_ctc_simulation.pdf`, `_out_fig6.json` |
| `figS1_persistence.py` | Persistence filtering, precision–recall | `fig_persistence.pdf`, `fig_precision_recall.pdf`, `_out_figS2.json` |
| `figS3_population.py` | Single-frequency probes | `fig_population.pdf`, `_out_figS3.json` |
| `fig7_empirical.py` | Literature values vs predictions (no simulation) | `fig_empirical_overlay.pdf` |
| `freqresp_sweep.py` | Gate + SFA frequency response over c and tau_w | `_out_freqresp_sweep.json` |
| `adaptation_chronaxie.py` | Chronaxie after threshold or conductance adaptation | `_out_adaptation_chronaxie.json` |
| `fig2_energy.py` | Stimulation energy | `fig2_energy_optimality.pdf` |

`noise_convergence.py` also writes convergence checks for some of these
analyses.

## Conventions

- Units: ms, mV, nA, MOhm (nA x MOhm = mV); HH block uses uA/cm^2, mS/cm^2.
- Gate: a spike requires an uninterrupted suprathreshold stretch of
  N = `round(c/dt)` samples and is stamped at the end of the step that
  completes the count, so the realised dwell lies between c - dt and c;
  after a spike v resets to 0 and the count is held at zero for `t_ref`,
  while v keeps integrating. c = 0 is the standard LIF rule.
- HH pulses end exactly at their nominal duration (the step containing the
  offset carries the fraction of the step inside the pulse).
- Response windows: *pulse-on* scores spikes before pulse offset (Eq. 8/9);
  *post-offset* also scores spikes after offset (Eq. 8b/9b).
- White current noise is an independent Gaussian sample per time step with
  s.d. sigma; its continuous-time intensity is sigma^2 * dt. When changing dt,
  scale sigma by sqrt(dt_old / dt_new) to keep the same noise process
  (`noise_convergence.py` does this).
- Seeds are fixed in each script (base seed 42; see scripts for per-figure
  schedules).

## License

MIT License (see `LICENSE`).

## Citation

If you use this code, please cite the accompanying article:

How much of chronaxie is spike-initiation dwell? Closed forms, a
Hodgkin–Huxley test and the limits of strength–duration measurements.
Manuscript submitted for publication.

The code used for the paper (v1.0.0) is archived on Zenodo:
https://doi.org/10.5281/zenodo.23599602. This section will be updated with
the journal reference once available.
