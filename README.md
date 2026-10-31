# CA3-inspired Hopfield model of memory deterioration

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
pytest              # 41 invariant tests (includes UI smoke tests; ~20 s)
```
Numba is used for the asynchronous update kernel; if it cannot be imported the code falls back to pure NumPy automatically.

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
