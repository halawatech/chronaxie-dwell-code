# Spike-Initiation Dwell Time — simulation code

Code and data for *Separating Membrane Charging from Spike-Initiation Dwell
in Chronaxie: A Persistence-Gated Neuron Model*.

## Environment

- Python 3.12.10, NumPy 2.4.6, SciPy 1.18.0, Matplotlib 3.10.9 (the versions
  used to generate the submitted figures; other recent versions should work).

## Installation and use

```bash
pip install -r requirements.txt
python fig3_duration.py
```

Each script runs from this folder, writes its figure (PDF and PNG) and, where
applicable, a JSON file with the numbers quoted in the paper. The `_out_*.json`
files in this repository are the outputs used for the submitted manuscript.

## Layout

| File | Purpose |
|---|---|
| `msmodels.py` | Shared models: Hodgkin–Huxley (per-gNa resting equilibrium, resting chord time constant, vectorised first-spike and threshold search), persistence-gated LIF (`lif_gate`), vectorised first-spike simulator (`first_spike_batch`), resonator impedance, OU noise, d50 estimation |
| `figstyle.py` | Shared Matplotlib style and colours, imported by every figure script. It sets appearance only (fonts, colours, line widths) and re-exports NumPy and Matplotlib; it computes nothing. It was rewritten because the original file was lost, and every submitted figure was generated with this version. |

## Script → figure map

Script numbers do **not** follow manuscript figure numbers.

| Script | Manuscript item | Output | Approx. runtime |
|---|---|---|---|
| `fig1_hh.py` | Fig. 1 (HH strength–duration curves, gate/RC/Weiss/EIF fits with a direct-inversion check of the EIF, half-step check, latency test) | `fig1_hh_chronaxie.pdf`, `_out_fig1.json` | ~20 min |
| `fig3_duration.py` | Fig. 2 (duration selectivity, pulse-on vs post-offset) | `fig3_duration_selectivity.pdf`, `_out_fig3.json` | ~2 min |
| `fig_identifiability.py` | Fig. 3 (parameter recovery: can c be identified?) | `fig_identifiability.pdf`, `_out_identifiability.json` | ~1 h; `--plot-only` redraws from the JSON |
| `fig4_bandpass.py` | Fig. 4 (rate ceiling) and Fig. S4 (frequency response, amplitude check) | `fig4_bandpass.pdf`, `fig_freq_response.pdf`, `_out_fig4.json` | ~15 min |
| `fig5_gatekeeper.py` | Fig. 5 (resonance gatekeeper) | `fig_gatekeeper.pdf`, `_out_fig5.json` | ~2 min |
| `fig6_ctc.py` | Fig. S8 (Supplementary Note: feedforward recruitment, 3 input realisations) | `fig_ctc_simulation.pdf`, `_out_fig6.json` | ~3 min |
| `figS5_realistic_taum.py` | Fig. S1 (decomposition across tau_m, both windows) | `fig_realistic_taum.pdf`, `_out_figS5.json` | ~5 min |
| `figS1_persistence.py` | Figs. S2 and S3 (persistence filtering, precision–recall) | `fig_persistence.pdf`, `fig_precision_recall.pdf`, `_out_figS2.json` | ~2 min |
| `figS3_population.py` | Fig. S5 (single-frequency probes) | `fig_population.pdf`, `_out_figS3.json` | ~1 min |
| `fig7_empirical.py` | Fig. S6 (literature vs predictions; no simulation) | `fig_empirical_overlay.pdf` | seconds |
| `figS4_precision.py` | Fig. S7 (latency vs jitter) | `fig_spike_precision.pdf`, `_out_figS4.json` | ~1 min |
| `noise_convergence.py` | Table S6 (time-step convergence at fixed noise intensity, including the small-gate results of Figs. 5 and S8; F1 credit-window sensitivity; jitter bootstrap intervals) | `_out_noise_convergence.json` | ~10 min |
| `freqresp_sweep.py` | Table S7 (gate + SFA frequency response over c and tau_w; direct sampling of each predicted centre and its neighbours) | `_out_freqresp_sweep.json` | ~40 min |
| `adaptation_chronaxie.py` | Section 5.1, Prediction P1 and Supplementary Methods (chronaxie after threshold or conductance adaptation that recovers during the test; asymptotic vs operational rheobase) | `_out_adaptation_chronaxie.json` | ~30 s |
| `hh_temperature.py` | Section 3.2 temperature paragraph, Table S8 and Supplementary Methods (HH chronaxie at 6.3–37 °C, two time steps, all-or-none check; full-curve fits of the reduction at 6.3, 15 and 20 °C) | `_out_hh_temperature.json` | ~45 min |

Runtimes are approximate, measured on an Intel Core i7-13700F (16 cores, 24 threads) with 32 GB RAM, one Python process per script.

Run each script from this folder, e.g. `python fig3_duration.py`. Each writes
its figure (PDF and PNG) and, where applicable, a JSON file with the numbers
quoted in the text.

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

`fig2_energy.py` (stimulation-energy figure) is kept for reference but is
no longer used in the manuscript.


## License

MIT License (see `LICENSE`).

## Citation

If you use this code, please cite the accompanying article:

Separating membrane charging from spike-initiation dwell in chronaxie: a
persistence-gated neuron model. Submitted to Biological Cybernetics.

The code as submitted (v1.0.0) is archived on Zenodo:
https://doi.org/10.5281/zenodo.23599602. This section will be updated with
the journal reference once available.
