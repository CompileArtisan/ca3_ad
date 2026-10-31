# IMPLEMENTATION_STATE

Actual public signatures (auto-generated from the code; regenerate after API changes).


## ca3model/config.py
```python
class DamageSpec
    def __init__(self, synaptic_severity: 'float' = 0.0, synaptic_mode: "Literal['deletion', 'attenuation']" = 'deletion', neuron_loss: 'float' = 0.0) -> None
    def to_dict(self) -> 'dict[str, Any]'
class ExperimentConfig
    def __init__(self, n_patterns: 'int' = 8, n_trials: 'int' = 30, base_seed: 'int' = 12345, n_new_patterns: 'int' = 5, plasticity_efficacy: 'float' = 1.0) -> None
    def to_dict(self) -> 'dict[str, Any]'
class NetworkConfig
    def __init__(self, n_neurons: 'int' = 256) -> None
    def to_dict(self) -> 'dict[str, Any]'
class RetrievalConfig
    def __init__(self, cue_noise: 'float' = 0.2, cue_mode: "Literal['flip', 'occlude']" = 'flip', max_sweeps: 'int' = 30, success_threshold: 'float' = 0.95) -> None
    def to_dict(self) -> 'dict[str, Any]'
```

## ca3model/rng.py
```python
class TrialRNGs
    def __init__(self, base_seed: 'int', trial: 'int', patterns: 'np.random.Generator', new_patterns: 'np.random.Generator', cues: 'np.random.Generator', damage: 'np.random.Generator', dynamics: 'np.random.Generator', children: 'tuple' = ()) -> None
    def fresh_cues(self) -> 'np.random.Generator'
    def fresh_dynamics(self) -> 'np.random.Generator'
def trial_rngs(base_seed: 'int', trial: 'int') -> 'TrialRNGs'
```

## ca3model/patterns.py
```python
def corrupt_cue(pattern: 'np.ndarray', noise: 'float', mode: 'str', rng: 'Generator') -> 'np.ndarray'
def grid_shape(n: 'int') -> 'tuple[int, int]'
def random_patterns(n_patterns: 'int', n_neurons: 'int', rng: 'Generator') -> 'np.ndarray'
def smooth_patterns(n_patterns: 'int', n_neurons: 'int', rng: 'Generator', length_scale: 'float' = 1.5) -> 'np.ndarray'
def to_image(vec: 'np.ndarray', shape: 'tuple[int, int] | None' = None) -> 'np.ndarray'
```

## ca3model/network.py
```python
class HopfieldNetwork
    def __init__(self, n: 'int') -> 'None'
    def connectivity_stats(self) -> 'dict'
    def copy(self) -> "'HopfieldNetwork'"
    def effective_weights(self) -> 'np.ndarray'
    def energy(self, state: 'np.ndarray') -> 'float'
    def store(self, patterns: 'np.ndarray', plasticity_efficacy: 'float' = 1.0) -> 'None'
```

## ca3model/damage.py
```python
class DamageDraw
    def __init__(self, U: 'np.ndarray', neuron_order: 'np.ndarray') -> None
def apply_damage(net: 'HopfieldNetwork', spec: 'DamageSpec', draw: 'DamageDraw') -> 'HopfieldNetwork'
def build_damage(draw: 'DamageDraw', spec: 'DamageSpec') -> 'tuple[np.ndarray, np.ndarray]'
def draw_damage_randomness(n: 'int', rng: 'Generator') -> 'DamageDraw'
def n_lost_neurons(n: 'int', fraction: 'float') -> 'int'
```

## ca3model/retrieval.py
```python
class RetrievalResult
    def __init__(self, final_state: 'np.ndarray', sweeps: 'int', n_flips: 'int', converged: 'bool', energy_trace: 'np.ndarray', state_history: 'list[np.ndarray] | None') -> None
def _async_kernel(W, state, alive, orders, max_sweeps)
def make_orders(alive: 'np.ndarray', max_sweeps: 'int', rng: 'Generator') -> 'np.ndarray'
def retrieve(net: 'HopfieldNetwork', cue: 'np.ndarray', max_sweeps: 'int', rng: 'Generator', record_states: 'bool' = False, W_eff: 'np.ndarray | None' = None, backend: 'str' = 'auto') -> 'RetrievalResult'
```

## ca3model/metrics.py
```python
def classify_attractor(state: 'np.ndarray', target: 'np.ndarray', alive: 'np.ndarray', patterns, success_threshold: 'float') -> 'str'
def evaluate_retrieval(result: 'RetrievalResult', target: 'np.ndarray', alive: 'np.ndarray', patterns, success_threshold: 'float') -> 'dict'
def hamming(state: 'np.ndarray', target: 'np.ndarray', mask: 'np.ndarray | None' = None) -> 'int'
def overlap(state: 'np.ndarray', target: 'np.ndarray', mask: 'np.ndarray | None' = None) -> 'float'
```

## ca3model/stats.py
```python
def mean_ci(x, confidence: 'float' = 0.95) -> 'tuple[float, float, float]'
def paired_difference(df: 'pd.DataFrame', cond_a: 'str', cond_b: 'str', value_col: 'str', condition_col: 'str' = 'condition') -> 'dict'
def paired_sweep_table(df: 'pd.DataFrame', treat: 'str', base: 'str', sweep_col: 'str', value_col: 'str') -> 'pd.DataFrame'
def summarise(df: 'pd.DataFrame', group_cols: 'list[str]', value_cols: 'list[str]') -> 'pd.DataFrame'
```

## ca3model/experiments.py
```python
def default_load_conditions(syn: 'float' = 0.3, neu: 'float' = 0.2, mode: 'str' = 'deletion') -> 'dict'
def default_load_values(n: 'int', steps: 'int' = 12) -> 'list[int]'
def exp_combined(net_cfg, exp_cfg, ret_cfg, syn_levels, neuron_levels, mode='deletion', progress: 'Progress' = None) -> 'pd.DataFrame'
def exp_healthy_vs_neuron_loss(net_cfg, exp_cfg, ret_cfg, fractions, mode='deletion', progress: 'Progress' = None) -> 'pd.DataFrame'
def exp_healthy_vs_weakened(net_cfg, exp_cfg, ret_cfg, severities, mode='deletion', progress: 'Progress' = None) -> 'pd.DataFrame'
def exp_memory_load(net_cfg, exp_cfg, ret_cfg, load_values=None, conditions=None, progress: 'Progress' = None, max_test_patterns: 'int' = 10) -> 'pd.DataFrame'
def exp_new_learning(net_cfg, exp_cfg, ret_cfg, severities, kind='synaptic', mode='deletion', progress: 'Progress' = None) -> 'pd.DataFrame'
def exp_weakening_vs_loss(net_cfg, exp_cfg, ret_cfg, levels, mode='deletion', progress: 'Progress' = None) -> 'pd.DataFrame'
def interaction_table(df: 'pd.DataFrame', value_col: 'str' = 'success') -> 'pd.DataFrame'
def matched_synapse_loss(f: 'float') -> 'float'
def run_condition_trial(net_healthy: 'HopfieldNetwork', damage_spec: 'DamageSpec', draw: 'DamageDraw', patterns_to_test: 'list[TestPattern]', ret_cfg: 'RetrievalConfig', rngs: 'TrialRNGs', label: 'str', extra: 'dict | None' = None, reported_spec: 'DamageSpec | None' = None) -> 'list[dict]'
def run_trial_conditions(net_cfg: 'NetworkConfig', exp_cfg: 'ExperimentConfig', ret_cfg: 'RetrievalConfig', conditions: 'dict[str, DamageSpec]', trial: 'int', experiment: 'str' = '', extra: 'dict | None' = None) -> 'list[dict]'
def sequential_learning_curve(net_cfg, exp_cfg, ret_cfg, spec: 'DamageSpec', progress: 'Progress' = None) -> 'pd.DataFrame'
```

## ca3model/plotting.py
```python
def color_for(name: 'str') -> 'str'
def fig_condition_bars(summary_df, y, label_col='condition', ylabel=None) -> 'go.Figure'
def fig_energy_trace(trace) -> 'go.Figure'
def fig_heatmap_grid(grid_df, x, y, z, zlabel=None, zmin=None, zmax=None, colorscale='Viridis') -> 'go.Figure'
def fig_memory_load(summary_df, n_neurons, x='n_patterns', y='success', color='condition', alpha_c=0.138) -> 'go.Figure'
def fig_metric_vs_severity(summary_df, x, y, color='condition', ylabel=None, xlabel='severity (model parameter)') -> 'go.Figure'
def fig_new_learning(summary_df, x='level', y='success', group_col='pattern_group') -> 'go.Figure'
def fig_pattern_panel(original, cue, healthy_recon, damaged_recon, shape=None, titles=('Original', 'Corrupted cue', 'Healthy reconstruction', 'Damaged reconstruction')) -> 'go.Figure'
def fig_weight_compare(W_healthy, W_damaged, alive) -> 'go.Figure'
def fig_weight_hist(W_healthy, W_damaged) -> 'go.Figure'
def fig_weight_matrix(W_eff, alive, title='Effective weights') -> 'go.Figure'
```


## Deviations / additions relative to the spec (all backwards compatible)
- `retrieve(..., W_eff=None, backend="auto")`: optional args to reuse effective weights and to force "numba"/"numpy" (needed for the Numba-vs-NumPy test).
- `retrieve` draws the update-order random block over **all N** neurons and then keeps the alive ones, so the random stream consumed and the relative order of survivors are identical across damage levels (paired design). Still `(max_sweeps, n_alive)` orders into the kernel.
- `_kernel_hist` is the real kernel (also writes per-sweep states); `_async_kernel` keeps the spec signature and wraps it. The NumPy fallback is the same function without compilation (`.py_func`).
- Zero-field tolerance: `|h| <= 1e-9*max|W|` counts as 0 (unchanged state) so float round-off cannot split ties differently between back-ends; scale-free.
- `TrialRNGs` additionally stores `base_seed`, `trial`, and offers `fresh_cues()` / `fresh_dynamics()`.
- `run_condition_trial(..., extra=None, reported_spec=None)`, `run_trial_conditions(..., experiment="", extra=None)`: optional args.
- Condition keys are `"name@level"`; text before `@` becomes the `condition` column.
- Suite D returns the tidy DataFrame; the interaction table comes from `interaction_table(df)`.
- Suite E adds columns `level`, `damage_kind`; `sequential_learning_curve(...)` is a separate function (adds `stage`, `step`).
- Suite C's `matched_synapse_loss_frac` = expected fraction of total synaptic efficacy removed (neuron: 1-(1-f)^2; synaptic: s for deletion, s/2 for attenuation).
- `HopfieldNetwork.connectivity_stats` also returns `frac_synapses_zero`.
- `stats.paired_sweep_table` is an extra helper.
