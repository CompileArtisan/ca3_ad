# RECALL4 NEUROLAB

A CA3-inspired model of memory deterioration.

Interactive, reproducible simulation of a Hopfield associative memory (inspired by hippocampal CA3 recurrent
collaterals) showing how two abstract forms of damage -- synaptic weakening and neuron loss -- affect
pattern completion from degraded cues and the learning of new patterns.

> This is a simplified computational abstraction inspired by hippocampal CA3 attractor dynamics. It does not reproduce Alzheimer's disease, does not model its molecular pathology, and must not be used to predict or interpret clinical cognitive decline in individuals. Results illustrate qualitative principles of attractor-network degradation only.

## Biology vs. computational abstraction

| Biological Alzheimer's pathology | What this model does instead (abstraction) |
|---|---|
| Synapse loss / synaptic dysfunction (amyloid-β oligomers, tau, reduced LTP) | Random, symmetric reduction of connection efficacy ("synaptic weakening": deletion or attenuation) |
| Neuron death, with regional vulnerability | Random removal of a fraction of units (silenced, cannot learn) |
| Impaired plasticity / encoding | Optional `plasticity_efficacy` scalar scaling new Hebbian increments |
| Amyloid plaques, tau tangles, inflammation, vascular factors, neuromodulators, genetics | **Not modelled** |
| CA3 as part of a trisynaptic circuit (DG, CA1, entorhinal input) | **Not modelled**; a single fully connected recurrent layer |
| Real memories are structured, sparse, graded, consolidated over time | Random dense bipolar patterns, one-shot Hebbian storage |


## Quick start
```bash
pip install -r requirements.txt
streamlit run app.py
pytest              # 68 tests (includes UI smoke tests; ~25 s)
```
Numba is used for the asynchronous update kernel; if it cannot be imported the code falls back to pure NumPy automatically.

## Simulator tab
The CA3 Hopfield reference model of the two Monte Carlo reports ("CA3 Hopfield Monte Carlo Experiment:
Connectivity Loss, Noise and Simulated Pharmacology" and "Experiment 2: Monte Carlo Validation of Donepezil- and
Memantine-Inspired Interventions", 18 Nov 2026), with their values and settings. Model in `ca3model/neurolab.py`.
- 25 neurons with states ±1; 3 random stored memories (seed 20261118), target = memory #1; Hebbian
  W = (1/N) Σ ξξᵀ, no self-connections.
- Cue: the same 5 neurons (2, 4, 6, 7, 11; seed 20261119) get memory #1's values, the other 20 start silent (0).
- **Functional connectivity** (20–100%) keeps exactly round(c × 300) of the 300 neuron pairs; removed pairs lose
  both w_ij and w_ji, survivors keep full strength. **Input noise** (0–50%) corrupts each neuron with probability p.
- Asynchronous updates in a fresh random order each sweep, s = sign(h), h = 0 keeps the state; at most 100 sweeps;
  success = settled with overlap m ≥ 0.8.
- Interventions (Experiment 2, default): **donepezil-inspired** = the 5 cue neurons at amplitude 1.25 / 1.50 / 1.75
  (effective cue 25 / 30 / 35%); **memantine-inspired** = input noise 2 / 4 / 6 points lower (10% → 8 / 6 / 4%).
  Experiment 1's version (cue gain g = 0.1, activity gate θ = 0.04) is in Advanced.
- The table under the animation repeats the settings over 2,000 matched trials (seed 8,000,000 + t), as in
  Experiment 2. At the reports' settings it lands within Monte Carlo error of their published numbers
  (`tests/test_neurolab.py` checks this).

## Layout
`ca3model/` model, experiments and Plotly builders (no Streamlit); `ui/` + `app.py` dashboard; `tests/` invariant tests;
`docs/` SPEC.md (contract), IMPLEMENTATION_STATE.md (actual signatures + deviations), LIMITATIONS.md.

## Reproducibility
Every random stream comes from `SeedSequence([base_seed, trial]).spawn(5)`; no global state. The same
`(base_seed, trial)` reproduces results exactly. Within a trial all conditions share patterns, cues, update orders and
damage randomness (paired design); damage is nested across severities.

## Conventions
Unit of replication = trial (patterns averaged within trial). Primary success = converged AND survivor accuracy >= threshold;
`success_all` also reported. Wilcoxon p-values are exploratory. Severity axes are model parameters, not disease stages.
