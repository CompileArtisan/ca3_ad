# PROJECT SPEC: CA3-Inspired Hopfield Model of Memory Deterioration

**Audience: the AI implementing the code. This file is the contract. Follow names, signatures, math and tests exactly.**

---

## 1. Purpose and scientific framing

Build an interactive, reproducible simulation of a Hopfield-style associative memory, inspired by hippocampal CA3 recurrent collaterals and pattern completion, and show how two abstract forms of damage affect (a) retrieval of stored memories from degraded cues and (b) learning of new patterns.

### 1.1 Biology vs. computational abstraction (must appear in the UI "About" tab and README)

| Biological Alzheimer's pathology | What this model does instead (abstraction) |
|---|---|
| Synapse loss / synaptic dysfunction (amyloid-β oligomers, tau, reduced LTP) | Random, symmetric reduction of connection efficacy ("synaptic weakening": deletion or attenuation) |
| Neuron death, with regional vulnerability | Random removal of a fraction of units (silenced, cannot learn) |
| Impaired plasticity / encoding | Optional `plasticity_efficacy` scalar scaling new Hebbian increments |
| Amyloid plaques, tau tangles, inflammation, vascular factors, neuromodulators, genetics | **Not modelled** |
| CA3 as part of a trisynaptic circuit (DG, CA1, entorhinal input) | **Not modelled**; a single fully connected recurrent layer |
| Real memories are structured, sparse, graded, consolidated over time | Random dense bipolar patterns, one-shot Hebbian storage |

**Mandatory disclaimer text** (verbatim, shown in app and README):
> "This is a simplified computational abstraction inspired by hippocampal CA3 attractor dynamics. It does not reproduce Alzheimer's disease, does not model its molecular pathology, and must not be used to predict or interpret clinical cognitive decline in individuals. Results illustrate qualitative principles of attractor-network degradation only."

Never use words such as "simulates Alzheimer's disease", "diagnoses", "predicts decline". Use "inspired by", "abstraction of", "analogue of".

### 1.2 Priorities
Mathematical transparency > reproducibility > interpretability > visual polish.

---

## 2. Technology choices

| Need | Tool | Reason |
|---|---|---|
| Language | Python 3.11/3.12 | Standard in computational neuroscience |
| Numerics | NumPy | Transparent linear algebra |
| Speed | Numba (`@njit`) for the asynchronous update kernel, **with a pure-NumPy fallback** selected automatically if Numba import fails | Async dynamics are sequential; pure Python is too slow for multi-trial sweeps |
| Results tables | pandas (tidy DataFrames) | Easy aggregation and CSV export |
| Stats | SciPy (`scipy.stats`) | t-intervals, Wilcoxon |
| Plots | Plotly (`plotly.express`, `graph_objects`) | Interactive, hover values, works natively in Streamlit |
| UI | Streamlit | Required by the brief |
| Tests | pytest | Invariant-based testing |

No other third-party dependencies. No deep-learning frameworks. No global random state.

---

## 3. Repository layout (exact)

```
ca3_ad_model/
├── app.py                      # Streamlit entry; thin, calls ui/ and ca3model/
├── requirements.txt
├── README.md
├── ca3model/
│   ├── __init__.py             # re-exports public API
│   ├── config.py               # frozen dataclasses
│   ├── rng.py                  # seeding utilities
│   ├── patterns.py             # pattern generation, cue corruption
│   ├── network.py              # HopfieldNetwork
│   ├── damage.py               # damage randomness + application
│   ├── retrieval.py            # asynchronous dynamics (Numba + fallback)
│   ├── metrics.py              # accuracy, Hamming, attractor classification
│   ├── stats.py                # CIs, paired comparisons
│   ├── experiments.py          # trial runner + experiment suites A–F
│   └── plotting.py             # Plotly figure builders (no Streamlit imports)
├── ui/
│   ├── __init__.py
│   ├── sidebar.py
│   ├── tabs.py                 # one render_* function per tab
│   └── text.py                 # disclaimer, help text, glossary
├── tests/
│   ├── test_patterns.py  test_network.py  test_damage.py
│   ├── test_retrieval.py test_metrics.py  test_stats.py
│   ├── test_experiments.py test_app_smoke.py
├── docs/
│   ├── SPEC.md                 # copy of this file
│   ├── IMPLEMENTATION_STATE.md # actual signatures, kept current
│   └── LIMITATIONS.md
└── results/                    # CSV exports (git-ignored)
```

Rules:
- Only `app.py` and `ui/` may import `streamlit`. `ca3model/` must be importable and testable without Streamlit.
- `plotting.py` returns `plotly.graph_objects.Figure`; it never displays.
- All public functions have type hints and docstrings that state the equation implemented.

---

## 4. Mathematical definitions (authoritative)

Notation: N neurons, P stored patterns, ξ^μ ∈ {−1,+1}^N.

**4.1 Storage (Hebbian, symmetric, zero self-connection)**
`W_raw = (1/N) Σ_μ ξ^μ (ξ^μ)ᵀ`, then `W_raw[i,i] = 0`. Incremental learning of a new pattern ξ with plasticity efficacy η ∈ [0,1]: `W_raw ← W_raw + (η/N)(ξξᵀ − I)`. (diag stays 0.)

**4.2 Damage state** (kept separate from learned weights)
- `efficacy` E ∈ [0,1]^{N×N}, symmetric, zero diagonal (healthy: 1 off-diagonal).
- `alive` ∈ {True,False}^N (healthy: all True).
- **Effective weights**: `W_eff[i,j] = W_raw[i,j] · E[i,j] · alive[i] · alive[j]`.
- Because damage is a separate mask, learning after damage automatically cannot restore deleted synapses or use dead neurons, while attenuated synapses learn with reduced efficacy.

**4.3 Synaptic weakening** with severity s ∈ [0,1] and a per-trial symmetric uniform matrix U (U_ij = U_ji ~ U(0,1), zero diagonal), drawn once per trial (common random numbers → damage is **nested**: larger s damages a superset):
- mode `"deletion"`: `E_ij = 0 if U_ij < s else 1` (fraction ≈ s of synapses removed).
- mode `"attenuation"`: `E_ij = 1 − s·U_ij` (every synapse weakened by a random amount; mean efficacy 1 − s/2).

> Teaching note (must be in docs and a unit test): **uniform** scaling `W ← c·W` (c>0) never changes retrieval under zero-threshold sign dynamics. Weakening only matters if it is heterogeneous. That is why the two modes above are random and per-synapse.

**4.4 Neuronal loss** with fraction f ∈ [0,1] and a per-trial random permutation π of neuron indices drawn once (nested): `k = round(f·N)`, `alive[π[:k]] = False`. Dead neurons: state fixed at 0, never updated, contribute nothing, excluded from "surviving" metrics.

**4.5 Combined damage**: apply both: `E_total = E_syn ⊙ outer(alive, alive)` (equivalent to 4.2). Order-independent.

**4.6 Dynamics (asynchronous, zero threshold)**
- Local field `h_i = Σ_j W_eff[i,j] s_j`.
- Update for alive neuron i in random order: `s_i ← +1 if h_i > 0, −1 if h_i < 0, unchanged if h_i == 0`.
- One **sweep** = one pass over all alive neurons in a fresh random permutation.
- Maintain `h` incrementally: when `s_i` flips by `Δ = s_new − s_old` (±2), `h += W_eff[:, i]·Δ`. This makes a sweep O(N + flips·N).
- **Converged** = a full sweep with zero flips. Stop at convergence or `max_sweeps`.
- **Convergence time** = number of sweeps until the first zero-flip sweep (so a cue that is already stable reports 1), plus total flips. If not converged, `sweeps = max_sweeps`, `converged = False`.
- Energy `E(s) = −½ sᵀ W_eff s`. With symmetric W_eff and zero diagonal it is non-increasing under async updates (unit test).

**4.7 Cue corruption**
- `"flip"`: flip exactly `round(c·N)` randomly chosen bits.
- `"occlude"`: choose `round(c·N)` bits and replace them with independent random ±1 (expected c/2 actually wrong; document this).
- Cue at dead neurons is set to 0 before dynamics.

**4.8 Metrics** (final state s, target ξ, alive mask a, n_alive = Σa)
- Overlap `m = (1/N_eval) Σ s_i ξ_i`.
- Hamming distance, **two versions**:
  - `hamming_all` = #{i : s_i ≠ ξ_i} over all N (dead neurons count as errors since s_i = 0 ≠ ξ_i).
  - `hamming_surv` = same restricted to alive neurons.
- `accuracy_all = 1 − hamming_all/N`; `accuracy_surv = 1 − hamming_surv/n_alive`. (Identity: accuracy = (1+m)/2 over the same set.)
- **Retrieval success (primary)**: `converged AND accuracy_surv ≥ success_threshold` (default 0.95). Also report `success_all` using `accuracy_all`. State in UI which is which. Primary uses survivors because the question is "can the remaining circuit complete the pattern"; `accuracy_all` shows the cost of lost units.
- **Attractor classification** among surviving neurons: compute overlap of s with each stored pattern and its negative. Return `attractor_type ∈ {"target","other_stored","mirror","spurious"}`: target if best match is the target with overlap ≥ threshold-equivalent; mirror if best match is −ξ of any stored pattern with |overlap| ≥ 0.9 and negative; spurious otherwise.

**4.9 Statistics**
- Unit of replication = **trial** (one random draw of patterns + damage + cues). Within a trial, average over test patterns first, then compute mean and 95% CI across trials (t-interval). Never treat individual patterns as independent replicates.
- Conditions in a trial share patterns, cues, update-order seeds and damage randomness (paired design). Comparisons report paired mean difference, 95% CI, and Wilcoxon signed-rank p-value, labelled *exploratory*. Success rates: trial-level fraction of successful patterns.
- Theory reference: random patterns have Hopfield capacity ≈ 0.138·N (α_c). Plot as a reference line; do not claim exact agreement at small N.

---

## 5. Seeding policy (`rng.py`)

```python
def trial_rngs(base_seed: int, trial: int) -> "TrialRNGs"  # dataclass of np.random.Generator
# fields: patterns, new_patterns, cues, damage, dynamics
```
Use `np.random.SeedSequence([base_seed, trial]).spawn(5)` → `default_rng` for each stream. Never use `np.random.seed`. Same `(base_seed, trial)` ⇒ identical results across runs and machines. Conditions inside a trial must **re-create** the `cues` and `dynamics` generators from the same child seed so that every condition sees identical cues and update orders.

---

## 6. Module contracts

### 6.1 `config.py` (all `@dataclass(frozen=True)`)
```python
NetworkConfig(n_neurons: int = 256)            # validated: 100 <= N <= 400
RetrievalConfig(cue_noise: float = 0.20, cue_mode: Literal["flip","occlude"] = "flip",
                max_sweeps: int = 30, success_threshold: float = 0.95)
DamageSpec(synaptic_severity: float = 0.0, synaptic_mode: Literal["deletion","attenuation"] = "deletion",
           neuron_loss: float = 0.0)           # validated ranges [0,1]
ExperimentConfig(n_patterns: int = 8, n_trials: int = 30, base_seed: int = 12345,
                 n_new_patterns: int = 5, plasticity_efficacy: float = 1.0)
```
Each has `.to_dict()` (JSON-safe) for caching keys, CSV metadata and reproducibility display.

### 6.2 `patterns.py`
```python
def random_patterns(n_patterns: int, n_neurons: int, rng: Generator) -> np.ndarray   # int8 (P,N) in {-1,+1}
def corrupt_cue(pattern: np.ndarray, noise: float, mode: str, rng: Generator) -> np.ndarray  # int8 (N,)
def to_image(vec: np.ndarray, shape: tuple[int,int] | None = None) -> np.ndarray     # (H,W); N must be square else nearest rectangle with padding NaN
def smooth_patterns(n_patterns, n_neurons, rng, length_scale=1.5) -> np.ndarray       # optional spatially smooth patterns, DEMO ONLY
```
`smooth_patterns` yields correlated patterns (lower capacity); UI must warn "visual demo only; statistics use random patterns".

### 6.3 `network.py`
```python
class HopfieldNetwork:
    n: int
    W_raw: np.ndarray         # float64 (N,N)
    efficacy: np.ndarray      # float64 (N,N)
    alive: np.ndarray         # bool (N,)
    n_stored: int
    patterns: list[np.ndarray]  # stored patterns (for diagnostics/attractor classification)
    @classmethod
    def from_patterns(cls, patterns: np.ndarray) -> "HopfieldNetwork"
    def store(self, patterns: np.ndarray, plasticity_efficacy: float = 1.0) -> None
    def effective_weights(self) -> np.ndarray             # per 4.2
    def copy(self) -> "HopfieldNetwork"
    def energy(self, state: np.ndarray) -> float
    def connectivity_stats(self) -> dict   # frac_synapses_lost (relative to healthy N(N-1) off-diag), n_alive, mean_efficacy_alive_pairs, mean_degree
```
Invariants: `W_raw` symmetric, diag 0; `efficacy` symmetric; `store` mutates in place and never touches `efficacy`/`alive`.

### 6.4 `damage.py`
```python
@dataclass(frozen=True)
class DamageDraw:  U: np.ndarray; neuron_order: np.ndarray
def draw_damage_randomness(n: int, rng: Generator) -> DamageDraw
def build_damage(draw: DamageDraw, spec: DamageSpec) -> tuple[np.ndarray, np.ndarray]   # (efficacy, alive)
def apply_damage(net: HopfieldNetwork, spec: DamageSpec, draw: DamageDraw) -> HopfieldNetwork
    # returns damaged COPY; new efficacy = net.efficacy * built_efficacy ; alive = net.alive & built_alive
```

### 6.5 `retrieval.py`
```python
@dataclass
class RetrievalResult:
    final_state: np.ndarray; sweeps: int; n_flips: int; converged: bool
    energy_trace: np.ndarray        # energy at start and after each sweep
    state_history: list[np.ndarray] | None   # only if record_states=True
def retrieve(net: HopfieldNetwork, cue: np.ndarray, max_sweeps: int, rng: Generator,
             record_states: bool = False) -> RetrievalResult
```
Numba kernel signature: `_async_kernel(W, state, alive, orders, max_sweeps) -> (state, sweeps, flips, converged)` where `orders` is a pre-generated `(max_sweeps, n_alive)` int array of permutations (generated with `rng` so results are deterministic and identical between Numba and fallback paths; test this). When `record_states` or energy trace needed, run sweep-by-sweep via the Python wrapper or have the kernel write per-sweep states into a preallocated array.

### 6.6 `metrics.py`
```python
def overlap(state, target, mask=None) -> float
def hamming(state, target, mask=None) -> int
def evaluate_retrieval(result, target, alive, patterns, success_threshold) -> dict
# keys: accuracy_all, accuracy_surv, hamming_all, hamming_surv, overlap_surv, sweeps, n_flips,
#       converged, success, success_all, attractor_type, final_energy
```

### 6.7 `stats.py`
```python
def mean_ci(x, confidence=0.95) -> tuple[float,float,float]          # mean, lo, hi (t-interval; handles n<2)
def summarise(df, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame
    # first mean over pattern_idx within (group_cols + trial), then mean/lo/hi across trials
def paired_difference(df, cond_a, cond_b, value_col) -> dict   # mean_diff, lo, hi, wilcoxon_p, n_trials
```

### 6.8 `experiments.py`
Common tidy-DataFrame columns (one row per trial × condition × test pattern):
`experiment, condition, trial, seed, pattern_idx, pattern_group ("old"/"new"/"all"), n_neurons, n_patterns, cue_noise, synaptic_mode, synaptic_severity, neuron_loss, plasticity_efficacy, accuracy_all, accuracy_surv, hamming_all, hamming_surv, sweeps, n_flips, converged, success, success_all, attractor_type, final_energy`

Core:
```python
def run_condition_trial(net_healthy, damage_spec, draw, patterns_to_test, ret_cfg, rngs, label) -> list[dict]
def run_trial_conditions(net_cfg, exp_cfg, ret_cfg, conditions: dict[str, DamageSpec], trial: int) -> list[dict]
```
Experiment suites (each returns a tidy DataFrame; each accepts `progress: Callable[[float], None] | None`):
- **A** `exp_healthy_vs_weakened(net_cfg, exp_cfg, ret_cfg, severities, mode)`: sweep s; neuron_loss = 0.
- **B** `exp_healthy_vs_neuron_loss(..., fractions)`: sweep f; s = 0.
- **C** `exp_weakening_vs_loss(..., levels)`: both swept on the same nominal axis **and** a "matched synapse loss" comparison: neuron loss f removes a fraction `1−(1−f)²` of synapses, so also show weakening at s = 1−(1−f)². Output a column `matched_synapse_loss_frac`.
- **D** `exp_combined(..., syn_levels, neuron_levels)`: full grid; conditions `healthy/synaptic_only/neuron_only/combined`; also return an `interaction` table: observed combined success vs. "independent" prediction `1 − (1−Δ_syn)−(1−Δ_neu)` style additive deficit (document formula: deficit_additive = deficit_syn + deficit_neu; interaction = deficit_combined − deficit_additive).
- **E** `exp_new_learning(..., severities, kind: {"synaptic","neuron","combined"})`:
  per trial: store `n_patterns` old patterns in healthy net; draw `n_new_patterns` new ones. Conditions:
  1. `healthy_learn`: healthy net learns the new patterns, test new + old.
  2. `damage_then_learn`: damage applied, then learn new patterns with `plasticity_efficacy`, test new + old.
  3. `learn_then_damage`: learn new patterns, then damage, test new + old (control: memory laid down before damage).
  Also `sequential_learning_curve`: add new patterns one at a time to a damaged vs healthy net, recording success of each new pattern right after learning and after all learning (interference).
  `pattern_group` ∈ {"old","new"}.
- **F** `exp_memory_load(..., load_values, conditions)`: sweep number of stored patterns P (default 1 → ≈0.25N in ~12 steps) for healthy/synaptic/neuron/combined at fixed severities; add column `load_alpha = P/N`.

Every suite stores `meta` in `df.attrs` (the `to_dict()` of all configs) for export.

### 6.9 `plotting.py` (all return `go.Figure`)
```python
fig_pattern_panel(original, cue, healthy_recon, damaged_recon, shape, titles)   # 1x4 heatmaps, black/white, lost neurons grey
fig_weight_matrix(W_eff, alive, title)                                          # heatmap, diverging colourscale centred on 0, lost rows/cols marked
fig_weight_compare(W_healthy, W_damaged, alive)                                 # side by side
fig_weight_hist(W_healthy, W_damaged)
fig_energy_trace(trace)
fig_metric_vs_severity(summary_df, x, y, color, ylabel)                         # line + CI ribbon
fig_condition_bars(summary_df, y)                                               # bars with CI error bars
fig_heatmap_grid(grid_df, x, y, z)                                              # combined damage heatmap
fig_new_learning(summary_df)                                                    # new vs old success, per condition
fig_memory_load(summary_df, n_neurons)                                          # with alpha_c = 0.138 reference line
```
Consistent colour map: healthy = blue, synaptic = orange, neuron loss = red, combined = purple. Colour-blind safe, axis labels with units, hover shows mean and CI.

---

## 7. Dashboard specification (Streamlit, `app.py` + `ui/`)

`st.set_page_config(layout="wide")`. Disclaimer banner (`st.info`) at the top on every page.

**Sidebar (global parameters, `ui/sidebar.py` returns config objects):**
- Network size N: select_slider over perfect squares {100,144,196,256,324,400}; default 256.
- Number of stored memories P (1–60, default 8) with live readout "load α = P/N vs α_c≈0.138".
- Cue noise (0–0.5), cue mode, max sweeps, success threshold.
- **Damage controls**: synaptic severity slider (0–1), synaptic mode radio, neuronal loss slider (0–0.9), plasticity efficacy slider (0–1; labelled "optional encoding impairment").
- Trials (5–200, default 30) and base seed (number_input) with "Re-roll seed" button.
- Pattern type: random (default) / smooth (demo-only warning).

**Tabs:**
1. **Memory demo**: choose stored memory index; panel of Original | Corrupted cue | Healthy reconstruction | Damaged reconstruction with metric cards (accuracy, Hamming, sweeps, success, attractor type). Energy trace plot; a "sweep" slider stepping through recorded states for the damaged network.
2. **Network connectivity**: healthy vs damaged weight-matrix heatmaps, lost neurons marked, weight histogram, connectivity stats (% synapses lost, n alive, mean effective degree).
3. **Damage sweeps** (Experiments A+B+C): run-button; plots accuracy, success rate, Hamming, convergence time vs severity for healthy baseline vs weakened vs neuron loss; matched-synapse-loss toggle; paired-difference table.
4. **Combined damage** (D): grid heatmap of success; selected-cell bar comparison healthy/syn/neuron/combined with CIs; interaction table with a plain-language note.
5. **New learning** (E): before/after damage learning performance for new and old patterns; sequential learning curve; plasticity-efficacy effect.
6. **Memory load** (F): success vs P for each condition, with α_c reference.
7. **Summary & export**: bar chart of all conditions at the current sidebar settings, results table, `st.download_button` CSV (including the config JSON as a header comment or companion file), and a reproducibility box (seed, N, versions of numpy/numba).
8. **About & limitations**: disclaimer, §1.1 table, glossary (attractor, Hebbian, pattern completion, Hamming distance, overlap), model assumptions, what the results do and do not show.

Behaviour rules:
- Cheap interactions (Memory demo, Network) recompute live. Heavy suites run only behind an `st.form` submit button, use `st.progress`, and are cached with `@st.cache_data` keyed by the JSON of all configs.
- Results persist in `st.session_state` so switching tabs does not rerun.
- No text in the UI claims clinical meaning. Axis titles say "severity (model parameter)", never "disease stage".
- Hover text on every control explaining it in one sentence.

---

## 8. Test invariants (`tests/`): invariants only, never expected scientific outcomes

1. Patterns are int8 in {−1,+1}, shape (P,N); same seed ⇒ identical; different seed ⇒ different.
2. `W_raw` symmetric, zero diagonal; equals `(1/N)ΣξξᵀminusDiag`; incremental `store` equals batch storage; `plasticity_efficacy=0` leaves W unchanged.
3. Low load (N=256, P=3): every stored pattern is a fixed point; retrieval from 10% flip noise gives accuracy 1.0.
4. Uniform scaling invariance: `retrieve` result identical for `W` and `5·W` given same cue/rng.
5. Energy trace non-increasing, including on damaged networks.
6. Numba path and NumPy fallback give identical final states, sweeps, flips for the same rng.
7. Damage: deletion fraction ≈ s (within binomial tolerance); `efficacy` symmetric, diag 0; nested (s1<s2 ⇒ deleted(s1) ⊆ deleted(s2)); neuron loss count == round(f·N) exactly; nested; combined damage independent of order; dead neurons end retrieval at 0 and never flip.
8. Metrics: `accuracy = 1 − hamming/N`; identity with overlap; `accuracy_all ≤ accuracy_surv` when any neuron dead (equal if none); mirror state classified `"mirror"`.
9. Stats: CI on known data; n=1 handled; `summarise` averages patterns within trial before across trials.
10. Experiments: determinism for same config; paired design (identical cue arrays across conditions within a trial); output has all required columns; severity 0 rows equal healthy rows exactly; learn-then-damage with severity 0 equals healthy_learn.
11. `test_app_smoke.py`: `streamlit.testing.v1.AppTest.from_file("app.py").run()` raises no exception.

Performance budget (must hold on a normal laptop with Numba): one retrieval at N=400 < 5 ms; an experiment suite with 30 trials × 8 patterns × 10 severities × 3 conditions < 30 s.

---

## 9. Defaults summary

| Parameter | Default | Range |
|---|---|---|
| N | 256 (16×16) | 100–400 |
| P | 8 | 1–60 |
| cue noise | 0.20 flip | 0–0.5 |
| max sweeps | 30 | 5–100 |
| success threshold (survivor accuracy) | 0.95 | 0.8–1.0 |
| trials | 30 | 5–200 |
| base seed | 12345 | any int |
| new patterns | 5 | 1–20 |
| plasticity efficacy | 1.0 | 0–1 |

## 10. Non-goals
No clinical claims; no real EEG/MRI/patient data; no biophysical neurons; no multi-region circuit; no training via backprop; no GPU; no user accounts or databases.

## 11. Output format rules for the implementing AI
- Provide each file complete, in its own code block, headed by its path.
- State any assumption or ambiguity you resolved, in ≤ 5 bullets, after the code.
- If you can execute code, run `pytest` for the stage's tests and report the real result; never claim tests pass without running them.
- Do not modify files outside the current stage unless told.
