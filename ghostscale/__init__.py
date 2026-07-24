"""Ghost Scale Simulation — active-inference model of the generative crash.

Package modules:
    constants          — tier / state / observation / action index names (Spec §2)
    config             — YAML -> Config loader with --quick and override support
    generative_model   — A, B, C, D construction + all construction-time assertions
    creators           — human creator agents (real POMDPs) and the synthetic generator
    environment        — signal-emission process (A1_generative) and corpus draws
    observer           — observer wrapper + rollout loop
    metrics            — formal metrics (EFE terms, entropies, MI, psi_analogue)
    figures            — shared plotting style + per-experiment figure builders
    experiments/       — E1..E6, each runnable standalone
"""
