from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


DEFAULT_MODEL_NUM_CELLS = {
    "AP": 32,
    "SP": 64,
    "AP_SP": 32,
    "TSNE_SP": 48,
    "TSNE_AP_SP": 64,
}

DEFAULT_MODEL_KERNEL_SIZE = {
    "AP": 4,
    "SP": 4,
    "AP_SP": 4,
    "TSNE_SP": 6,
    "TSNE_AP_SP": 6,
}

DEFAULT_REPLAY_STRATEGIES = [
    "random",
    "ensemble_mi",
    "similarity_penalized_mean",
    "family_qbc",
    "cluster_diverse_representative",
    "oed_logdet",
    "hybrid_mi_diverse",
]

THESIS_FULL_REPLAY_STRATEGIES = [
    "random",
    "ensemble_mean",
    "similarity_penalized_mean",
    "predictive_entropy",
    "ensemble_mi",
    "ucb",
    "family_qbc",
    "cluster_diverse_representative",
    "oed_logdet",
    "hybrid_mi_diverse",
]

DEFAULT_DISCOVERY_STRATEGIES = [
    "ensemble_mean",
    "ucb",
    "ei",
    "pi",
    "mes",
]

GENERATOR_OBJECTIVE_MODES = [
    "fixed_mean",
    "match_acquisition",
    "broad_pool",
    "bo_utility",
]

BINARY_THRESHOLD_STRATEGIES = [
    "pr_best_f1",
    "fixed_0_5",
]


@dataclass
class RunConfig:
    run_name: str = "default_run"
    output_root: str = "active_learning_runs"
    random_seed: int = 20260317
    holdout_fraction: float = 0.20
    validation_fraction_of_development: float = 0.20
    replay_seed_size: int = 40
    batch_size: int = 5
    max_rounds: int = 10
    candidate_pool_min: int = 50
    ga_max_attempts: int = 5
    ensemble_size: int = 5
    ensemble_seeds: list[int] = field(
        default_factory=lambda: [11, 23, 37, 53, 71]
    )
    epochs: int = 70
    real_strategy: str = "ensemble_mi"
    replay_strategies: list[str] = field(
        default_factory=lambda: list(DEFAULT_REPLAY_STRATEGIES)
    )
    discovery_strategies: list[str] = field(
        default_factory=lambda: list(DEFAULT_DISCOVERY_STRATEGIES)
    )
    discovery_export_count: int | None = None
    discovery_ucb_beta: float = 1.0
    discovery_improvement_xi: float = 0.0
    discovery_mes_samples: int = 128
    generator_objective_mode: str = "match_acquisition"
    use_similarity_penalty: bool = False
    use_length_penalty: bool = True
    binary_threshold_strategy: str = "pr_best_f1"
    use_calibrated_acquisition: bool = True
    calibration_l2: float = 1e-3
    calibration_learning_rate: float = 0.05
    calibration_max_iter: int = 500
    allowed_amino_acids: str = "ACDEFGHIKLMNPQRSTVWY"
    preferred_length_min: int = 5
    preferred_length_max: int = 10
    min_initial_peptide_length: int = 3
    max_initial_peptide_length: int = 24
    population_size: int = 50
    offspring_count: int = 30
    max_num_generations: int = 30
    tournament_size: int = 3
    mutation_probability: float = 0.05
    diversity_prefilter_multiplier: int = 3
    oed_regularization: float = 1e-3
    persist_replay_ledgers: bool = True
    train_family_for_init: bool = False
    phase: str = ""
    branch_strategy: str = ""
    backup_strategy: str = ""
    phase1_root: str = ""
    phase2_root: str = ""
    phase3_output_root: str = ""
    md_inventory_root: str = ""
    supek_queue: str = "gpu"
    supek_ncpus: int = 4
    supek_ngpus: int = 1
    supek_mem: str = "40GB"
    supek_walltime: str = ""
    model_num_cells: dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_MODEL_NUM_CELLS)
    )
    model_kernel_size: dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_MODEL_KERNEL_SIZE)
    )

    def __post_init__(self) -> None:
        if len(self.ensemble_seeds) < self.ensemble_size:
            base = self.ensemble_seeds[-1] if self.ensemble_seeds else 0
            while len(self.ensemble_seeds) < self.ensemble_size:
                base += 17
                self.ensemble_seeds.append(base)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.discovery_export_count is None:
            self.discovery_export_count = self.batch_size
        if self.discovery_export_count <= 0:
            raise ValueError("discovery_export_count must be positive")
        if self.replay_seed_size <= 0:
            raise ValueError("replay_seed_size must be positive")
        if self.discovery_mes_samples <= 0:
            raise ValueError("discovery_mes_samples must be positive")
        if self.generator_objective_mode not in GENERATOR_OBJECTIVE_MODES:
            allowed = ", ".join(GENERATOR_OBJECTIVE_MODES)
            raise ValueError(
                f"generator_objective_mode must be one of: {allowed}"
            )
        if self.binary_threshold_strategy not in BINARY_THRESHOLD_STRATEGIES:
            allowed = ", ".join(BINARY_THRESHOLD_STRATEGIES)
            raise ValueError(
                f"binary_threshold_strategy must be one of: {allowed}"
            )
        if self.calibration_l2 < 0:
            raise ValueError("calibration_l2 must be non-negative")
        if self.calibration_learning_rate <= 0:
            raise ValueError("calibration_learning_rate must be positive")
        if self.calibration_max_iter <= 0:
            raise ValueError("calibration_max_iter must be positive")
        if self.holdout_fraction <= 0 or self.holdout_fraction >= 1:
            raise ValueError("holdout_fraction must be in (0, 1)")
        if (
            self.validation_fraction_of_development <= 0
            or self.validation_fraction_of_development >= 1
        ):
            raise ValueError(
                "validation_fraction_of_development must be in (0, 1)"
            )

    @property
    def run_dir(self) -> Path:
        return Path(self.output_root) / self.run_name

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "RunConfig":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "generator_objective_mode" not in payload:
            payload["generator_objective_mode"] = "fixed_mean"
        if "use_similarity_penalty" not in payload:
            payload["use_similarity_penalty"] = True
        if "use_length_penalty" not in payload:
            payload["use_length_penalty"] = True
        if "binary_threshold_strategy" not in payload:
            payload["binary_threshold_strategy"] = "fixed_0_5"
        return cls(**payload)
