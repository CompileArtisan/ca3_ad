"""CA3-inspired Hopfield model of memory deterioration (computational abstraction)."""
from .config import DamageSpec, ExperimentConfig, NetworkConfig, RetrievalConfig
from .damage import DamageDraw, apply_damage, build_damage, draw_damage_randomness
from .experiments import (exp_combined, exp_healthy_vs_neuron_loss, exp_healthy_vs_weakened,
                          exp_memory_load, exp_new_learning, exp_weakening_vs_loss,
                          interaction_table, run_condition_trial, run_trial_conditions,
                          sequential_learning_curve)
from .metrics import evaluate_retrieval, hamming, overlap
from .network import HopfieldNetwork
from .patterns import corrupt_cue, random_patterns, smooth_patterns, to_image
from .retrieval import HAVE_NUMBA, RetrievalResult, retrieve
from .rng import TrialRNGs, trial_rngs
from .stats import mean_ci, paired_difference, paired_sweep_table, summarise

ALPHA_C = 0.138  # theoretical Hopfield capacity for random patterns (reference only)
