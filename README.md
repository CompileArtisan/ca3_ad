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
One brain, one stored memory (an X on a 5x5 grid of 25 model neurons). Model in `ca3model/neurolab.py`.
- **Alzheimer's severity** (0% = healthy) weakens every synapse by a random amount (up to 90% of synaptic
  strength gone at 100%) and removes up to 2 neurons. Damage is nested: a higher severity is the same brain worse.
- Neurons fire only when recurrent input beats a firing threshold, so weaker synapses mean a cue delivers less
  drive. A healthy brain completes the X from a 40% cue; a moderately damaged one often cannot.
  (The analysis tabs use zero-threshold sign dynamics, where uniform weakening never changes recall; that is
  why the old simulator showed a damaged brain recalling perfectly.)
- **Cue strength** = share of the X's 9 neurons lit; **input noise** = share of the other 16 lit by mistake.
- **Donepezil** restores 50% of lost synaptic strength; **memantine** halves input noise (the slider moves).

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
