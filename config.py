"""Trial configuration, constants, and queue builders."""

from dataclasses import dataclass

# ── Experiment constants ──────────────────────────────────────────────
N_AGENTS = 10
N_ROUNDS = 5
MAX_TOKENS = 512
HISTORY_WINDOW = 2

TOPOLOGIES = ["fc", "chain", "circle", "star"]
MINORITY_RATIOS = [0.1, 0.2, 0.3]

# Position configs per topology.
#   chain: 0=head, 1=middle(N//2), 2=tail(N-1)
#   star:  0=minority as leaf, 1=minority as hub
#   fc/circle: single config (positions equivalent)
POSITION_CONFIGS = {
    "fc": [0],
    "chain": [0, 1, 2],
    "circle": [0],
    "star": [0, 1],
}


@dataclass
class TrialConfig:
    scenario_id: str
    topology: str
    minority_ratio: float
    position_config: int
    temperature: float
    seed: int
    model_condition: str  # "prompt_induced" or "model_induced"

    @property
    def trial_id(self) -> str:
        return (
            f"{self.scenario_id}_{self.topology}_{self.minority_ratio}"
            f"_pos{self.position_config}_t{self.temperature}"
            f"_s{self.seed}_{self.model_condition}"
        )


def build_primary_queue(scenarios: list[dict]) -> list[TrialConfig]:
    """1,050 trials: 4 topos x 3 ratios x position ablations x 50 scenarios."""
    trials = []
    for s in scenarios:
        for topo in TOPOLOGIES:
            for ratio in MINORITY_RATIOS:
                for pos in POSITION_CONFIGS[topo]:
                    trials.append(TrialConfig(
                        scenario_id=s["id"],
                        topology=topo,
                        minority_ratio=ratio,
                        position_config=pos,
                        temperature=0.7,
                        seed=42,
                        model_condition="prompt_induced",
                    ))
    return trials


def build_primary_em_queue(scenarios: list[dict]) -> list[TrialConfig]:
    """1,050 trials: same grid as primary but with model_induced (EM) condition."""
    trials = []
    for s in scenarios:
        for topo in TOPOLOGIES:
            for ratio in MINORITY_RATIOS:
                for pos in POSITION_CONFIGS[topo]:
                    trials.append(TrialConfig(
                        scenario_id=s["id"],
                        topology=topo,
                        minority_ratio=ratio,
                        position_config=pos,
                        temperature=0.7,
                        seed=42,
                        model_condition="model_induced",
                    ))
    return trials


def build_ablation_queue(
    scenarios: list[dict], ablation: str
) -> list[TrialConfig]:
    """50 trials each: FC, ratio=0.2, pos=0.

    ablation: one of "t0", "0.5b", "14b", "model_induced"
    """
    temp = 0.0 if ablation == "t0" else 0.7
    condition = "model_induced" if ablation == "model_induced" else "prompt_induced"
    seed = 0 if ablation == "t0" else 42
    return [
        TrialConfig(
            scenario_id=s["id"],
            topology="fc",
            minority_ratio=0.2,
            position_config=0,
            temperature=temp,
            seed=seed,
            model_condition=condition,
        )
        for s in scenarios
    ]


def build_seed_replication(
    scenarios: list[dict],
    conditions: list[dict],
    seeds: list[int] | None = None,
) -> list[TrialConfig]:
    """Targeted replication: 2 extra seeds on conditions with signal.

    conditions: list of dicts with keys
        {topology, minority_ratio, position_config, model_condition}
    """
    if seeds is None:
        seeds = [123, 456]
    trials = []
    for cond in conditions:
        for s in scenarios:
            for seed in seeds:
                trials.append(TrialConfig(
                    scenario_id=s["id"],
                    topology=cond["topology"],
                    minority_ratio=cond["minority_ratio"],
                    position_config=cond["position_config"],
                    temperature=0.7,
                    seed=seed,
                    model_condition=cond["model_condition"],
                ))
    return trials
