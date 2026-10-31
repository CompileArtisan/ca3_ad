"""All user-facing text: disclaimer, help strings, glossary, assumptions."""

DISCLAIMER = (
    "This is a simplified computational abstraction inspired by hippocampal CA3 attractor dynamics. "
    "It does not reproduce Alzheimer's disease, does not model its molecular pathology, and must not be "
    "used to predict or interpret clinical cognitive decline in individuals. Results illustrate "
    "qualitative principles of attractor-network degradation only."
)

BIO_TABLE = """
| Biological Alzheimer's pathology | What this model does instead (abstraction) |
|---|---|
| Synapse loss / synaptic dysfunction (amyloid-β oligomers, tau, reduced LTP) | Random, symmetric reduction of connection efficacy ("synaptic weakening": deletion or attenuation) |
| Neuron death, with regional vulnerability | Random removal of a fraction of units (silenced, cannot learn) |
| Impaired plasticity / encoding | Optional `plasticity_efficacy` scalar scaling new Hebbian increments |
| Amyloid plaques, tau tangles, inflammation, vascular factors, neuromodulators, genetics | **Not modelled** |
| CA3 as part of a trisynaptic circuit (DG, CA1, entorhinal input) | **Not modelled**; a single fully connected recurrent layer |
| Real memories are structured, sparse, graded, consolidated over time | Random dense bipolar patterns, one-shot Hebbian storage |
"""

GLOSSARY = {
    "Attractor": "A stable network state the dynamics settle into. Stored memories are (ideally) attractors.",
    "Hebbian learning": "Connections between units that are active together are strengthened: W = (1/N) Σ ξ ξᵀ.",
    "Pattern completion": "Recovering a full stored pattern from a partial or noisy cue.",
    "Hamming distance": "Number of units whose state differs from the target pattern.",
    "Overlap": "m = (1/N) Σ sᵢξᵢ; +1 = identical, 0 = unrelated, −1 = mirror image. Accuracy = (1+m)/2.",
    "Sweep": "One asynchronous pass updating every surviving unit once, in random order.",
    "Spurious state": "A stable state that is not any stored pattern (e.g. a mixture of several).",
    "Mirror state": "The exact negative −ξ of a stored pattern; it is also an attractor in Hopfield networks.",
    "Load α": "Stored patterns per neuron, α = P/N. Random patterns overload the network near α_c ≈ 0.138.",
}

ASSUMPTIONS = """
- Random dense ±1 patterns, one-shot Hebbian storage, a single fully connected layer, zero self-connections.
- Damage is a separate mask (efficacy per synapse, alive per neuron) applied to learned weights; it is nested
  across severities (common random numbers) and paired across conditions inside each trial.
- Uniform down-scaling of all weights never changes retrieval (zero-threshold sign dynamics); weakening matters
  only because it is *heterogeneous*.
- With plasticity efficacy = 1, learning-then-damage and damage-then-learning give identical effective weights
  in this abstraction; they differ only when plasticity efficacy < 1.
- Replication unit = trial; patterns are averaged within a trial before computing CIs. p-values are exploratory.
"""

SHOW_NOT_SHOW = """
**What the results can show:** qualitative principles — attractor basins shrink, convergence slows, spurious or
mirror states appear, capacity falls, and neuron loss costs more than its raw synapse count suggests because
both endpoints of a synapse must survive.

**What they do not show:** anything about disease progression, staging, individual risk, biomarkers, or
treatment. Severity axes are model parameters, not disease stages.
"""

HELP = {
    "n": "Number of model neurons (units) in the recurrent layer.",
    "p": "How many random patterns are stored with the Hebbian rule.",
    "noise": "Fraction of cue bits corrupted before retrieval starts.",
    "mode_cue": "'flip' negates exactly that fraction; 'occlude' replaces bits with random ±1 (about half end up wrong).",
    "sweeps": "Maximum asynchronous sweeps before retrieval is declared not converged.",
    "thr": "Retrieval counts as successful if the final state is stable and survivor accuracy reaches this value.",
    "syn": "Fraction of synapses removed (deletion) or the maximum random attenuation (attenuation).",
    "syn_mode": "Deletion zeros random synapses; attenuation multiplies every synapse by a random factor in [1−s, 1].",
    "neu": "Fraction of neurons silenced; lost neurons cannot learn or take part in retrieval.",
    "eta": "Optional encoding impairment: scales every new Hebbian increment (1 = normal learning).",
    "trials": "Independent random draws of patterns, damage and cues; confidence intervals use trials as replicates.",
    "seed": "Base random seed; the same seed reproduces all results exactly.",
    "ptype": "'random' is used for all statistics; 'smooth' gives visually structured patterns for demos only.",
    "new": "Number of new patterns learned in the learning experiments.",
}
