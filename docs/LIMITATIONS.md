# LIMITATIONS

This is a simplified computational abstraction inspired by hippocampal CA3 attractor dynamics. It does not reproduce Alzheimer's disease, does not model its molecular pathology, and must not be used to predict or interpret clinical cognitive decline in individuals. Results illustrate qualitative principles of attractor-network degradation only.

## Model limitations
- Single fully connected recurrent layer; no dentate gyrus, CA1, entorhinal input, or inhibition.
- Random dense bipolar patterns; real memories are sparse, graded and correlated. `smooth` patterns are a visual demo only.
- One-shot Hebbian storage; no consolidation, forgetting, or time course.
- Damage is random and uniform: no regional vulnerability, no clustering of synapse/neuron loss, no spreading pathology.
- Severity axes are model parameters, **not** disease stages; nothing is calibrated to data.
- Small networks (N<=400): the theoretical capacity alpha_c ~ 0.138 holds only for large N; finite-size agreement is approximate.

## Teaching notes
- Uniform scaling of all weights (W -> cW, c>0) never changes zero-threshold sign dynamics; weakening matters only because it is heterogeneous (unit-tested).
- Under deletion, a diluted Hopfield network stays robust until severe dilution; the sharp failure appears at high severity. Attenuation is gentler still (mean efficacy 1 - s/2).
- Neuron loss of fraction f silences 1-(1-f)^2 of synapses; compare with `matched_synaptic` in the sweep tab.
- Because W_eff = W_raw * efficacy * alive, damage-then-learn and learn-then-damage give identical effective weights when plasticity efficacy = 1; they differ only for efficacy < 1.
- `success` (survivors) asks whether the remaining circuit completes the pattern; `success_all` charges the loss of neurons as errors. They answer different questions.

## Statistics
- Replication unit = trial; patterns are averaged within trial first. Confidence intervals are t-intervals over trials; Wilcoxon p-values are exploratory and uncorrected for multiple comparisons.

## Forbidden interpretations
This model must not be used to diagnose, stage, predict, or interpret cognitive decline in any individual.
