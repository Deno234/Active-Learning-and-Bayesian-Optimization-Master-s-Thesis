from __future__ import annotations

import argparse
import json
from pathlib import Path

from active_learning_thesis.config import (
    BINARY_THRESHOLD_STRATEGIES,
    GENERATOR_OBJECTIVE_MODES,
    RunConfig,
)
from active_learning_thesis.dashboard_integrity import main as dashboard_integrity_main
from active_learning_thesis.dashboard_feedback import run_feedback_loop
from active_learning_thesis.dashboard_md_slate import run_md_slate_supervisor
from active_learning_thesis.dashboard_run_setup import dashboard_init_run, parse_strategy_list
from active_learning_thesis.dashboard_smoke import main as dashboard_smoke_main
from active_learning_thesis.dashboard import launch_dashboard
from active_learning_thesis.bura_autopilot import recover_bura_outputs_only, run_bura_full_autopilot
from active_learning_thesis.bura_md_workflow import run_bura_md_workflow_from_payload
from active_learning_thesis.md_orchestrator import (
    finalize_md_stage,
    md_ladder_status,
    prepare_manual_md_stage,
    prepare_md_stage,
)
from active_learning_thesis.md_workflow import (
    PROFILE_CHOICES,
    build_pdbs,
    make_md_ingest_csv,
    parse_bura_md_benchmark,
    parse_md_results,
    prepare_bura_md_benchmark,
    prepare_md_campaign,
)
from active_learning_thesis.optional_evaluator_study import run_optional_evaluator_study
from active_learning_thesis.phase1_reproduction import format_status_report, run_phase1_reproduce
from active_learning_thesis.phase2_replay import (
    format_phase2_status,
    run_phase2_export,
    run_phase2_replay,
)
from active_learning_thesis.phase3_strategy_selection import run_phase3_strategy_selection
from active_learning_thesis.phase3_real_al import (
    DEFAULT_STRATEGIES as PHASE3_REAL_AL_DEFAULT_STRATEGIES,
    run_phase3_real_al,
)
from active_learning_thesis.phase4_bo import (
    PHASE4_POLICIES,
    run_phase4_bo,
)
from active_learning_thesis.phase4_diversity import (
    init_phase4_diversity,
    run_phase4_diversity,
)
from active_learning_thesis.phase5_self_paced import (
    PHASE5_STRATEGIES,
    run_phase5,
)
from active_learning_thesis.study import compare_studies, run_study, summarize_study
from active_learning_thesis.thesis_canary import run_thesis_canary
from active_learning_thesis.thesis_figures import build_thesis_figures
from active_learning_thesis.thesis_freeze import freeze_final_result
from active_learning_thesis.thesis_packet import export_thesis_packet
from active_learning_thesis.workflow import (
    evaluate_final,
    ingest_round,
    init_run,
    propose_round,
    run_discovery,
    run_replay,
)


def _add_generator_controls(parser: argparse.ArgumentParser) -> None:
    defaults = RunConfig()
    parser.add_argument(
        "--generator-objective-mode",
        choices=GENERATOR_OBJECTIVE_MODES,
        default=defaults.generator_objective_mode,
        help=(
            "How GA candidate generation is tied to acquisition. "
            "match_acquisition is research-clean; fixed_mean preserves the "
            "older practical discovery behavior."
        ),
    )
    parser.add_argument(
        "--use-similarity-penalty",
        action="store_true",
        default=defaults.use_similarity_penalty,
        help="Apply GA similarity shaping during candidate generation.",
    )
    parser.add_argument(
        "--no-length-penalty",
        action="store_true",
        help="Disable peptide length shaping during candidate generation.",
    )


def _add_threshold_controls(parser: argparse.ArgumentParser) -> None:
    defaults = RunConfig()
    parser.add_argument(
        "--binary-threshold-strategy",
        choices=BINARY_THRESHOLD_STRATEGIES,
        default=defaults.binary_threshold_strategy,
        help=(
            "Binary evaluation threshold. New thesis runs default to "
            "PR-best-F1 thresholded F1; fixed_0_5 preserves the old cutoff."
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Active-learning thesis workflow for peptide self-assembly.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser(
        "init-run",
        help="Create a run directory, split manifest, ledger, and baseline models.",
    )
    init_parser.add_argument("--run-name", required=True)
    init_parser.add_argument("--output-root", default="active_learning_runs")
    init_parser.add_argument("--random-seed", type=int, default=20260317)
    init_parser.add_argument("--batch-size", type=int, default=5)
    init_parser.add_argument("--max-rounds", type=int, default=10)
    init_parser.add_argument("--epochs", type=int, default=70)
    init_parser.add_argument("--candidate-pool-min", type=int, default=50)
    init_parser.add_argument("--replay-seed-size", type=int, default=40)
    init_parser.add_argument("--real-strategy", default="ensemble_mi")
    init_parser.add_argument(
        "--replay-strategies",
        nargs="*",
        default=None,
        help="Override the default replay benchmark strategies.",
    )
    init_parser.add_argument(
        "--train-family-for-init",
        action="store_true",
        help="Also train the family committee during init-run.",
    )
    init_parser.add_argument(
        "--raw-acquisition",
        action="store_true",
        help="Use raw, uncalibrated model probabilities for acquisition and metrics in this run.",
    )
    _add_generator_controls(init_parser)
    _add_threshold_controls(init_parser)

    replay_parser = subparsers.add_parser(
        "run-replay",
        help="Run the offline replay benchmark over the configured strategies.",
    )
    replay_parser.add_argument("--run-dir", required=True)
    replay_parser.add_argument(
        "--strategies",
        nargs="*",
        default=None,
        help="Replay strategies to run. Defaults to the run config.",
    )

    study_parser = subparsers.add_parser(
        "summarize-study",
        help="Aggregate replay evidence across runs into thesis-ready study tables.",
    )
    study_parser.add_argument("--run-root", default="active_learning_runs")
    study_parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for study CSV/JSON outputs. Defaults to <run-root>/_study_evidence.",
    )
    study_parser.add_argument(
        "--metric",
        default="f1",
        help="Metric to aggregate from replay curves, e.g. f1, pr_auc, roc_auc, balanced_accuracy, brier_score, log_loss, ece_10.",
    )
    study_parser.add_argument(
        "--target",
        type=float,
        default=None,
        help="Optional target metric value for label-efficiency summaries.",
    )
    study_parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Optional run directory names to include. Defaults to all replay-capable runs under run-root.",
    )

    run_study_parser = subparsers.add_parser(
        "run-study",
        help="Run a resumable multi-seed replay benchmark study and summarize the evidence.",
    )
    run_study_parser.add_argument("--study-name", required=True)
    run_study_parser.add_argument("--run-root", default="active_learning_runs")
    run_study_parser.add_argument("--seeds", type=int, default=5, help="Number of study seeds to run.")
    run_study_parser.add_argument("--seed-start", type=int, default=20260317)
    run_study_parser.add_argument("--seed-step", type=int, default=1009)
    run_study_parser.add_argument(
        "--seed-index-start",
        type=int,
        default=1,
        help="One-based display/run-name index for the first seed. Useful when splitting one study across per-seed jobs.",
    )
    run_study_parser.add_argument("--epochs", type=int, default=70)
    run_study_parser.add_argument("--max-rounds", type=int, default=10)
    run_study_parser.add_argument("--batch-size", type=int, default=5)
    run_study_parser.add_argument("--candidate-pool-min", type=int, default=50)
    run_study_parser.add_argument("--replay-seed-size", type=int, default=40)
    run_study_parser.add_argument("--real-strategy", default="ensemble_mi")
    run_study_parser.add_argument("--ensemble-size", type=int, default=5)
    run_study_parser.add_argument(
        "--strategies",
        nargs="*",
        default=None,
        help="Replay strategies to benchmark. Defaults to the RunConfig defaults.",
    )
    run_study_parser.add_argument("--metric", default="f1")
    run_study_parser.add_argument("--target", type=float, default=None)
    run_study_parser.add_argument("--train-family-for-init", action="store_true")
    run_study_parser.add_argument(
        "--raw-acquisition",
        action="store_true",
        help="Use raw, uncalibrated model probabilities for acquisition and reporting metrics.",
    )
    _add_generator_controls(run_study_parser)
    _add_threshold_controls(run_study_parser)
    run_study_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the study manifest without launching model training.",
    )
    run_study_parser.add_argument(
        "--force-replay",
        action="store_true",
        help="Rerun replay strategies even when summary artifacts already exist.",
    )
    run_study_parser.add_argument(
        "--no-summarize",
        action="store_true",
        help="Skip final summarize-study aggregation.",
    )
    run_study_parser.add_argument(
        "--allow-config-mismatch",
        action="store_true",
        help="Resume existing run directories even if their config differs from this study plan.",
    )

    compare_study_parser = subparsers.add_parser(
        "compare-studies",
        help="Compare two matched study manifests, e.g. calibrated vs raw acquisition.",
    )
    compare_study_parser.add_argument("--run-root", default="active_learning_runs")
    compare_study_parser.add_argument(
        "--baseline-study",
        required=True,
        help="Baseline study name, manifest path, or study directory.",
    )
    compare_study_parser.add_argument(
        "--candidate-study",
        required=True,
        help="Candidate study name, manifest path, or study directory.",
    )
    compare_study_parser.add_argument("--metric", default="f1")
    compare_study_parser.add_argument("--target", type=float, default=None)
    compare_study_parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for comparison outputs. Defaults to <run-root>/_studies/_comparisons/...",
    )

    propose_parser = subparsers.add_parser(
        "propose-round",
        help="Generate and score a candidate pool, then export a CG-MD batch.",
    )
    propose_parser.add_argument("--run-dir", required=True)
    propose_parser.add_argument(
        "--strategy",
        default=None,
        help="Override the configured real active-learning strategy for this round.",
    )

    ingest_parser = subparsers.add_parser(
        "ingest-round",
        help="Import CG-MD labels for the pending batch and retrain/evaluate the model.",
    )
    ingest_parser.add_argument("--run-dir", required=True)
    ingest_parser.add_argument("--import-csv", required=True)

    discovery_parser = subparsers.add_parser(
        "run-discovery",
        help="Run BO-style discovery strategies on the latest trained ensemble.",
    )
    discovery_parser.add_argument("--run-dir", required=True)
    discovery_parser.add_argument(
        "--strategies",
        nargs="*",
        default=None,
        help="Discovery strategies to run. Defaults to the run config.",
    )

    final_eval_parser = subparsers.add_parser(
        "evaluate-final",
        help="Evaluate the latest trained ensemble once on the frozen holdout.",
    )
    final_eval_parser.add_argument("--run-dir", required=True)

    freeze_final_parser = subparsers.add_parser(
        "freeze-final",
        help="Freeze the final thesis result with consistency checks, fingerprints, and a model card.",
    )
    freeze_final_parser.add_argument("--run-dir", required=True)
    freeze_final_parser.add_argument("--metric", default="f1")
    freeze_final_parser.add_argument(
        "--run-evaluation",
        action="store_true",
        help="Run evaluate-final before freezing. By default, an existing metrics/final_holdout.json is required.",
    )
    freeze_final_parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing final_freeze/final_freeze.json.",
    )
    freeze_final_parser.add_argument(
        "--allow-unresolved",
        action="store_true",
        help="Write a frozen_with_unresolved report instead of treating blocker checks as fatal.",
    )
    freeze_final_parser.add_argument("--json", action="store_true")

    prepare_md_parser = subparsers.add_parser(
        "prepare-md-campaign",
        help="Prepare a BURA-safe MD campaign from an exported CG-MD batch.",
    )
    prepare_md_parser.add_argument("--run-dir", required=True)
    prepare_md_parser.add_argument("--batch-csv", required=True)
    prepare_md_parser.add_argument("--campaign", required=True)
    prepare_md_parser.add_argument("--cluster", default="bura")
    prepare_md_parser.add_argument("--md-profile", required=True, choices=PROFILE_CHOICES)

    build_pdb_parser = subparsers.add_parser(
        "build-pdbs",
        help="Build PDBs locally with PyMOL or validate manually provided PDBs.",
    )
    build_pdb_parser.add_argument("--campaign-dir", required=True)
    build_pdb_parser.add_argument("--validate-only", action="store_true")

    parse_md_parser = subparsers.add_parser(
        "parse-md-results",
        help="Parse finished MD outputs into md_review.csv.",
    )
    parse_md_parser.add_argument("--campaign-dir", required=True)

    ingest_md_parser = subparsers.add_parser(
        "make-md-ingest-csv",
        help="Convert a reviewed md_review.csv into the ingest-round CSV schema.",
    )
    ingest_md_parser.add_argument("--campaign-dir", required=True)
    ingest_md_parser.add_argument("--review-csv", required=True)

    prepare_md_stage_parser = subparsers.add_parser(
        "prepare-md-stage",
        help="Prepare a guided single-peptide MD campaign and local staging assets.",
    )
    prepare_md_stage_parser.add_argument("--run-dir", required=True)
    prepare_md_stage_parser.add_argument("--batch-csv", required=True)
    prepare_md_stage_parser.add_argument("--sequence", required=True)
    prepare_md_stage_parser.add_argument("--campaign", required=True)
    prepare_md_stage_parser.add_argument("--md-profile", required=True, choices=PROFILE_CHOICES)
    prepare_md_stage_parser.add_argument("--cluster", default="bura")
    prepare_md_stage_parser.add_argument("--reuse-pdb-from", default=None)
    prepare_md_stage_parser.add_argument("--exclude-nodes", default="")

    prepare_manual_md_stage_parser = subparsers.add_parser(
        "prepare-manual-md-stage",
        help="Prepare an MD campaign for a manually chosen peptide outside the AL loop.",
    )
    prepare_manual_md_stage_parser.add_argument("--run-dir", required=True)
    prepare_manual_md_stage_parser.add_argument("--sequence", required=True)
    prepare_manual_md_stage_parser.add_argument("--campaign", required=True)
    prepare_manual_md_stage_parser.add_argument("--md-profile", required=True, choices=PROFILE_CHOICES)
    prepare_manual_md_stage_parser.add_argument("--cluster", default="bura")
    prepare_manual_md_stage_parser.add_argument("--reuse-pdb-from", default=None)
    prepare_manual_md_stage_parser.add_argument("--exclude-nodes", default="")

    finalize_md_stage_parser = subparsers.add_parser(
        "finalize-md-stage",
        help="Parse a finished guided MD campaign and print the next recommended step.",
    )
    finalize_md_stage_parser.add_argument("--campaign-dir", required=True)
    finalize_md_stage_parser.add_argument("--staged-package-dir", default=None)

    bura_autopilot_parser = subparsers.add_parser(
        "bura-full-autopilot",
        help="Run one full BURA campaign from upload through local parse/finalize.",
    )
    bura_autopilot_parser.add_argument("--run-root", required=True)
    bura_autopilot_parser.add_argument("--campaign-dir", required=True)
    bura_autopilot_parser.add_argument("--sequence", required=True)
    bura_autopilot_parser.add_argument("--exclude-nodes", default="")
    bura_autopilot_parser.add_argument("--poll-seconds", type=int, default=60)
    bura_autopilot_parser.add_argument("--max-wait-seconds", type=int, default=60 * 60 * 24 * 14)

    bura_recovery_parser = subparsers.add_parser(
        "bura-recover-md-outputs",
        help="Copy back and finalize an existing BURA campaign without submitting a new job.",
    )
    bura_recovery_parser.add_argument("--run-root", required=True)
    bura_recovery_parser.add_argument("--campaign-dir", required=True)
    bura_recovery_parser.add_argument("--sequence", required=True)

    bura_md_workflow_parser = subparsers.add_parser(
        "dashboard-run-bura-md-workflow",
        help="Run the Dash BURA MD macro workflow from a serialized payload.",
    )
    bura_md_workflow_parser.add_argument("--payload-json", required=True)

    prepare_bura_benchmark_parser = subparsers.add_parser(
        "prepare-bura-md-benchmark",
        help="Prepare an optional BURA MD performance benchmark from an equilibrated package.",
    )
    prepare_bura_benchmark_parser.add_argument("--campaign-dir", required=True)
    prepare_bura_benchmark_parser.add_argument("--sequence", required=True)
    prepare_bura_benchmark_parser.add_argument("--benchmark-name", default="bura_perf")
    prepare_bura_benchmark_parser.add_argument("--nsteps", type=int, default=50000)
    prepare_bura_benchmark_parser.add_argument(
        "--layouts",
        default="",
        help="Comma-separated layout names. Defaults to the built-in safe BURA benchmark set.",
    )
    prepare_bura_benchmark_parser.add_argument("--walltime", default="02:00:00")

    parse_bura_benchmark_parser = subparsers.add_parser(
        "parse-bura-md-benchmark",
        help="Parse an optional BURA MD performance benchmark into benchmark_results.csv.",
    )
    parse_bura_benchmark_parser.add_argument("--benchmark-dir", required=True)

    ladder_status_parser = subparsers.add_parser(
        "md-ladder-status",
        help="Show guided MD ladder status for one peptide within a run.",
    )
    ladder_status_parser.add_argument("--run-dir", required=True)
    ladder_status_parser.add_argument("--sequence", required=True)

    dashboard_parser = subparsers.add_parser(
        "dashboard",
        help="Launch the local Streamlit admin dashboard over the run directory tree.",
    )
    dashboard_parser.add_argument("--run-root", default="active_learning_runs")
    dashboard_parser.add_argument("--host", default="127.0.0.1")
    dashboard_parser.add_argument("--port", type=int, default=8501)
    dashboard_parser.add_argument("--refresh-seconds", type=int, default=0)
    dashboard_parser.add_argument("--skip-integrity-check", action="store_true")

    dashboard_check_parser = subparsers.add_parser(
        "dashboard-check",
        help="Run dashboard integrity smoke checks without launching Streamlit.",
    )
    dashboard_check_parser.add_argument("--run-root", default="active_learning_runs")
    dashboard_check_parser.add_argument("--refresh-seconds", type=int, default=0)
    dashboard_check_parser.add_argument("--json", action="store_true")

    dashboard_smoke_parser = subparsers.add_parser(
        "dashboard-smoke",
        help="Run the dashboard integrity matrix plus the key dashboard-focused test suites.",
    )
    dashboard_smoke_parser.add_argument("--run-root", default="active_learning_runs")
    dashboard_smoke_parser.add_argument("--refresh-seconds", type=int, default=0)
    dashboard_smoke_parser.add_argument("--json", action="store_true")
    dashboard_smoke_parser.add_argument(
        "--tests",
        nargs="*",
        default=None,
        help="Optional unittest module names to run instead of the default dashboard smoke suites.",
    )

    thesis_canary_parser = subparsers.add_parser(
        "thesis-canary",
        help="Run a deterministic synthetic end-to-end thesis loop canary.",
    )
    thesis_canary_parser.add_argument("--run-root", default="active_learning_runs")
    thesis_canary_parser.add_argument("--name", default="seeded_thesis_canary")
    thesis_canary_parser.add_argument("--seed", type=int, default=20260425)
    thesis_canary_parser.add_argument("--peptides", type=int, default=2)
    thesis_canary_parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing canary run with the same name and seed under <run-root>/_thesis_canaries.",
    )
    thesis_canary_parser.add_argument("--json", action="store_true")

    thesis_packet_parser = subparsers.add_parser(
        "export-thesis-packet",
        help="Export thesis-ready evidence tables, canary status, and reproducibility metadata.",
    )
    thesis_packet_parser.add_argument("--run-root", default="active_learning_runs")
    thesis_packet_parser.add_argument("--output-dir", default=None)
    thesis_packet_parser.add_argument("--title", default="thesis_packet")
    thesis_packet_parser.add_argument("--metric", default="f1")
    thesis_packet_parser.add_argument(
        "--skip-dashboard",
        action="store_true",
        help="Skip dashboard-derived lifecycle/readiness tables and export only disk-scanned artifacts.",
    )
    thesis_packet_parser.add_argument("--json", action="store_true")

    thesis_figures_parser = subparsers.add_parser(
        "build-thesis-figures",
        help="Build output-only thesis SVG figures, clean tables, and captions from an exported thesis packet.",
    )
    thesis_figures_parser.add_argument("--packet-dir", required=True)
    thesis_figures_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <packet-dir>/thesis_figures.",
    )
    thesis_figures_parser.add_argument(
        "--metric",
        default=None,
        help="Metric to plot. Defaults to the packet metric.",
    )
    thesis_figures_parser.add_argument("--json", action="store_true")

    optional_evaluator_parser = subparsers.add_parser(
        "optional-evaluator-study",
        help="Run an optional sidecar analysis of external evaluator disagreement and peptide complexity bins.",
    )
    optional_evaluator_parser.add_argument("--run-dir", required=True)
    optional_evaluator_parser.add_argument(
        "--external-scores",
        default=None,
        help="Optional CSV with sequence plus score/external_score/probability and optional evaluator columns.",
    )
    optional_evaluator_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <run-dir>/_optional_evaluator_study.",
    )
    optional_evaluator_parser.add_argument("--bin-count", type=int, default=4)
    optional_evaluator_parser.add_argument("--json", action="store_true")

    md_slate_runner_parser = subparsers.add_parser(
        "dashboard-run-md-slate",
        help=argparse.SUPPRESS,
    )
    md_slate_runner_parser.add_argument("--run-root", required=True)
    md_slate_runner_parser.add_argument("--slate-id", required=True)

    feedback_runner_parser = subparsers.add_parser(
        "dashboard-continue-feedback",
        help=argparse.SUPPRESS,
    )
    feedback_runner_parser.add_argument("--run-dir", required=True)
    feedback_runner_parser.add_argument("--propose-next-batch", action="store_true")

    setup_runner_parser = subparsers.add_parser(
        "dashboard-init-run",
        help=argparse.SUPPRESS,
    )
    setup_runner_parser.add_argument("--run-root", required=True)
    setup_runner_parser.add_argument("--run-name", required=True)
    setup_runner_parser.add_argument("--random-seed", type=int, required=True)
    setup_runner_parser.add_argument("--batch-size", type=int, required=True)
    setup_runner_parser.add_argument("--max-rounds", type=int, required=True)
    setup_runner_parser.add_argument("--epochs", type=int, required=True)
    setup_runner_parser.add_argument("--candidate-pool-min", type=int, required=True)
    setup_runner_parser.add_argument("--replay-seed-size", type=int, required=True)
    setup_runner_parser.add_argument("--real-strategy", required=True)
    setup_runner_parser.add_argument("--replay-strategies", default="")
    setup_runner_parser.add_argument("--train-family-for-init", action="store_true")
    setup_runner_parser.add_argument("--raw-acquisition", action="store_true")
    _add_generator_controls(setup_runner_parser)
    _add_threshold_controls(setup_runner_parser)
    setup_runner_parser.add_argument("--pin-run", action="store_true")
    setup_runner_parser.add_argument("--run-label", default="")
    setup_runner_parser.add_argument("--skip-baseline-init", action="store_true")
    setup_runner_parser.add_argument("--run-replay-after-init", action="store_true")

    phase1_parser = subparsers.add_parser(
        "phase1-reproduce",
        help="Build the Phase 1 baseline reproduction package without running active-learning replay.",
    )
    phase1_parser.add_argument("--output-root", default="thesis_results/01_reproduction")
    phase1_parser.add_argument("--dataset-path", default="SA_ML_predictive/data/data_SA.csv")
    phase1_parser.add_argument("--mode", choices=["full"], default=None)
    phase1_parser.add_argument(
        "--step",
        choices=["sanity", "folds", "nested-cv", "aggregate-nested-cv", "thresholds", "train-final", "generate", "cgmd-template", "checklist"],
        default=None,
    )
    phase1_parser.add_argument("--status", action="store_true")
    phase1_parser.add_argument("--dry-run", action="store_true")
    phase1_parser.add_argument("--force", action="store_true")
    phase1_parser.add_argument("--skip-heavy", action="store_true")
    phase1_parser.add_argument("--write-supek-pbs", action="store_true")
    phase1_parser.add_argument(
        "--pbs-repo-root",
        default=None,
        help=(
            "Absolute SUPEK repository root to bake into generated PBS stdout/stderr "
            "and command paths. Use this when generating PBS scripts from a different machine."
        ),
    )
    phase1_parser.add_argument("--models", nargs="*", default=None)
    phase1_parser.add_argument("--model", default=None, help="Single-model shortcut for --models.")
    phase1_parser.add_argument("--seed", type=int, default=20260317)
    phase1_parser.add_argument("--epochs", type=int, default=70)
    phase1_parser.add_argument(
        "--generation-target-unique",
        type=int,
        default=20,
        help=(
            "Number of unique AP_SP GA candidates to export in Phase 1. "
            "Use 50 to mirror the paper GA population size, or 5 for a top-five thesis table."
        ),
    )
    phase1_parser.add_argument(
        "--generation-minimum-return-count",
        type=int,
        default=10,
        help=(
            "Minimum number of generated candidates accepted before failing. "
            "Set equal to --generation-target-unique for strict paper-population matching."
        ),
    )
    phase1_parser.add_argument(
        "--generation-ga-max-attempts",
        type=int,
        default=20,
        help="Maximum retry attempts for Phase 1 GA candidate generation.",
    )

    phase2_parser = subparsers.add_parser(
        "phase2-replay",
        help="Run Phase 2 manifest-driven active-learning replay experiments.",
    )
    phase2_parser.add_argument("--mode", choices=["smoke", "ablation", "benchmark", "aggregate"], default=None)
    phase2_parser.add_argument("--phase1-root", default="thesis_results/01_reproduction")
    phase2_parser.add_argument("--output-root", default="thesis_results/02_replay")
    phase2_parser.add_argument("--status", action="store_true")
    phase2_parser.add_argument("--force", action="store_true")
    phase2_parser.add_argument("--write-supek-pbs", action="store_true")
    phase2_parser.add_argument(
        "--pbs-repo-root",
        default=None,
        help="Absolute SUPEK repository root to bake into generated PBS paths.",
    )
    phase2_parser.add_argument("--outer-folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    phase2_parser.add_argument("--inner-fold", type=int, default=1)
    phase2_parser.add_argument("--replay-seed-sizes", nargs="+", type=int, default=[10, 40])
    phase2_parser.add_argument("--batch-size", type=int, default=5)
    phase2_parser.add_argument("--max-rounds", type=int, default=20)
    phase2_parser.add_argument("--strategies", nargs="*", default=None)
    phase2_parser.add_argument("--setup", default=None)
    phase2_parser.add_argument("--ensemble-size", type=int, default=5)
    phase2_parser.add_argument("--calibrated", action="store_true")
    phase2_parser.add_argument("--base-seed", type=int, default=20260317)

    phase2_export_parser = subparsers.add_parser(
        "phase2-export",
        help="Export Phase 2 thesis-ready evidence tables and static figures.",
    )
    phase2_export_parser.add_argument("--input-root", default="thesis_results/02_replay")
    phase2_export_parser.add_argument("--output-root", default="thesis_results/02_replay/evidence")

    phase3_select_parser = subparsers.add_parser(
        "phase3-select-strategies",
        help="Recommend Phase 3 Real AL strategies from Phase 2 replay outputs.",
    )
    phase3_select_parser.add_argument("--phase2-root", default="thesis_results/02_replay")
    phase3_select_parser.add_argument("--output-root", default="thesis_results/03_real_al_strategy_selection")
    phase3_select_parser.add_argument("--top-k", type=int, default=3)
    phase3_select_parser.add_argument("--exclude", nargs="*", default=["random"])
    phase3_select_parser.add_argument("--min-overlap-warning", type=float, default=0.40)

    phase3_real_parser = subparsers.add_parser(
        "phase3-real-al",
        help="Initialize and operate isolated Phase 3 Real AL strategy branches.",
    )
    phase3_real_subparsers = phase3_real_parser.add_subparsers(
        dest="phase3_real_al_action",
        required=True,
    )

    def add_phase3_common(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--output-root", default="thesis_results/03_real_al")

    def add_supek_controls(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--pbs-repo-root", default=None)
        parser.add_argument("--supek-walltime", default=None)
        parser.add_argument("--supek-queue", default=None)
        parser.add_argument("--supek-ncpus", type=int, default=None)
        parser.add_argument("--supek-ngpus", type=int, default=None)
        parser.add_argument("--supek-mem", default=None)

    phase3_init_parser = phase3_real_subparsers.add_parser(
        "init",
        help="Create isolated Phase 3 Real AL branches and SUPEK proposal previews.",
    )
    add_phase3_common(phase3_init_parser)
    add_supek_controls(phase3_init_parser)
    phase3_init_parser.add_argument("--phase1-root", default="thesis_results/01_reproduction")
    phase3_init_parser.add_argument("--phase2-root", default="thesis_results/02_replay")
    phase3_init_parser.add_argument("--strategies", nargs="+", default=list(PHASE3_REAL_AL_DEFAULT_STRATEGIES))
    phase3_init_parser.add_argument("--backup-strategy", default="ensemble_mi")
    phase3_init_parser.add_argument("--force", action="store_true")
    phase3_init_parser.add_argument("--random-seed", type=int, default=20260317)
    phase3_init_parser.add_argument("--replay-seed-size", type=int, default=40)
    phase3_init_parser.add_argument("--batch-size", type=int, default=5)
    phase3_init_parser.add_argument("--max-rounds", type=int, default=10)
    phase3_init_parser.add_argument("--candidate-pool-min", type=int, default=50)
    phase3_init_parser.add_argument("--ga-max-attempts", type=int, default=100)
    phase3_init_parser.add_argument("--ensemble-size", type=int, default=5)
    phase3_init_parser.add_argument("--epochs", type=int, default=70)
    phase3_init_parser.add_argument("--raw-acquisition", action="store_true")
    phase3_init_parser.add_argument("--generator-objective-mode", choices=GENERATOR_OBJECTIVE_MODES, default="match_acquisition")
    phase3_init_parser.add_argument("--use-similarity-penalty", action="store_true", default=True)
    phase3_init_parser.add_argument("--no-length-penalty", action="store_true")
    phase3_init_parser.add_argument("--binary-threshold-strategy", choices=BINARY_THRESHOLD_STRATEGIES, default="pr_best_f1")

    phase3_propose_parser = phase3_real_subparsers.add_parser(
        "propose",
        help="Run or preview one Phase 3 branch proposal round.",
    )
    add_phase3_common(phase3_propose_parser)
    add_supek_controls(phase3_propose_parser)
    phase3_propose_parser.add_argument("--branch", required=True)
    phase3_propose_parser.add_argument("--round", type=int, default=1)
    phase3_propose_parser.add_argument("--dry-run", action="store_true")
    phase3_propose_parser.add_argument("--write-supek-pbs", action="store_true")

    phase3_compare_parser = phase3_real_subparsers.add_parser(
        "compare",
        help="Write read-only Phase 3 branch comparison reports.",
    )
    add_phase3_common(phase3_compare_parser)
    phase3_compare_parser.add_argument("--round", type=int, default=1)

    phase3_status_parser = phase3_real_subparsers.add_parser(
        "status",
        help="Summarize Phase 3 Real AL branch and MD inventory state.",
    )
    add_phase3_common(phase3_status_parser)

    phase3_make_ingest_parser = phase3_real_subparsers.add_parser(
        "make-ingest-csv",
        help="Create a branch-local Phase 3 ingest CSV from reviewed MD evidence.",
    )
    add_phase3_common(phase3_make_ingest_parser)
    phase3_make_ingest_parser.add_argument("--branch", required=True)
    phase3_make_ingest_parser.add_argument("--round", type=int, default=1)
    phase3_make_ingest_parser.add_argument("--force", action="store_true")

    phase3_ingest_parser = phase3_real_subparsers.add_parser(
        "ingest",
        help="Ingest reviewed Phase 3 labels into one branch ledger only.",
    )
    add_phase3_common(phase3_ingest_parser)
    phase3_ingest_parser.add_argument("--branch", required=True)
    phase3_ingest_parser.add_argument("--round", type=int, default=1)
    phase3_ingest_parser.add_argument("--import-csv", required=True)
    phase3_ingest_parser.add_argument("--dry-run", action="store_true")
    phase3_ingest_parser.add_argument("--force", action="store_true")

    phase3_finalize_parser = phase3_real_subparsers.add_parser(
        "finalize",
        help="Retrain a fully ingested Phase 3 branch and optionally evaluate the frozen holdout.",
    )
    add_phase3_common(phase3_finalize_parser)
    add_supek_controls(phase3_finalize_parser)
    phase3_finalize_parser.add_argument("--branch", required=True)
    phase3_finalize_parser.add_argument("--round", type=int, required=True)
    phase3_finalize_parser.add_argument("--evaluate-holdout", action="store_true")
    phase3_finalize_parser.add_argument("--dry-run", action="store_true")
    phase3_finalize_parser.add_argument("--write-supek-pbs", action="store_true")
    phase3_finalize_parser.add_argument("--force", action="store_true")

    phase4_parser = subparsers.add_parser(
        "phase4-bo",
        help="Initialize and run one-round Phase 4 BO-guided generative discovery.",
    )
    phase4_subparsers = phase4_parser.add_subparsers(
        dest="phase4_bo_action",
        required=True,
    )

    def add_phase4_common(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--output-root",
            default="thesis_results/04_bayesian_optimization",
        )

    phase4_init_parser = phase4_subparsers.add_parser(
        "init",
        help="Validate canonical data and write Phase 4 manifests and SUPEK previews.",
    )
    add_phase4_common(phase4_init_parser)
    phase4_init_parser.add_argument(
        "--phase1-root",
        default="thesis_results/01_reproduction",
    )
    phase4_init_parser.add_argument(
        "--phase3-root",
        default="thesis_results/03_real_al",
    )
    phase4_init_parser.add_argument(
        "--policies",
        nargs="+",
        default=list(PHASE4_POLICIES),
    )
    phase4_init_parser.add_argument("--round", type=int, default=1)
    phase4_init_parser.add_argument("--random-seed", type=int, default=20260317)
    phase4_init_parser.add_argument("--ensemble-size", type=int, default=5)
    phase4_init_parser.add_argument("--epochs", type=int, default=70)
    phase4_init_parser.add_argument("--kappa", type=float, default=1.0)
    phase4_init_parser.add_argument("--xi", type=float, default=0.0)
    phase4_init_parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-8,
    )
    phase4_init_parser.add_argument("--pbs-repo-root", default=None)
    phase4_init_parser.add_argument("--supek-queue", default="gpu")
    phase4_init_parser.add_argument("--supek-ncpus", type=int, default=4)
    phase4_init_parser.add_argument("--supek-ngpus", type=int, default=1)
    phase4_init_parser.add_argument("--supek-mem", default="40GB")
    phase4_init_parser.add_argument("--force", action="store_true")

    phase4_train_parser = phase4_subparsers.add_parser(
        "train-ensemble",
        help="Train the fixed-split Phase 4 ensemble with Phase 3 member-wise calibration.",
    )
    add_phase4_common(phase4_train_parser)
    phase4_train_parser.add_argument("--force", action="store_true")

    phase4_propose_parser = phase4_subparsers.add_parser(
        "propose",
        help="Run or preview one Phase 4 proposal policy.",
    )
    add_phase4_common(phase4_propose_parser)
    phase4_propose_parser.add_argument("--branch", choices=PHASE4_POLICIES, required=True)
    phase4_propose_parser.add_argument("--round", type=int, default=1)
    phase4_propose_parser.add_argument("--dry-run", action="store_true")
    phase4_propose_parser.add_argument("--write-supek-pbs", action="store_true")
    phase4_propose_parser.add_argument("--pbs-repo-root", default=None)
    phase4_propose_parser.add_argument("--force", action="store_true")

    phase4_compare_parser = phase4_subparsers.add_parser(
        "compare",
        help="Write descriptive Phase 4 comparison and branch-status reports.",
    )
    add_phase4_common(phase4_compare_parser)
    phase4_compare_parser.add_argument("--round", type=int, default=1)

    phase4_status_parser = phase4_subparsers.add_parser(
        "status",
        help="Summarize Phase 4 model, policy, and proposal state.",
    )
    add_phase4_common(phase4_status_parser)

    phase4d_init_parser = phase4_subparsers.add_parser(
        "init-diversity-aware",
        help="Initialize the secondary Phase 4-D fresh generative replicate.",
    )
    add_phase4_common(phase4d_init_parser)
    phase4d_init_parser.add_argument("--source-round", type=int, default=1)
    phase4d_init_parser.add_argument("--phase4d-run", type=int, default=1)
    phase4d_init_parser.add_argument("--seed-config", default=None)
    phase4d_init_parser.add_argument("--phase4d-walltime", default=None)
    phase4d_init_parser.add_argument("--pbs-repo-root", default=None)
    phase4d_init_parser.add_argument("--force", action="store_true")

    phase4d_run_parser = phase4_subparsers.add_parser(
        "run-diversity-aware",
        help="Run the resumable Phase 4-D fresh generation and paired selectors.",
    )
    add_phase4_common(phase4d_run_parser)
    phase4d_run_parser.add_argument("--source-round", type=int, default=1)
    phase4d_run_parser.add_argument("--phase4d-run", type=int, default=1)
    phase4d_run_parser.add_argument("--seed-config", default=None)

    phase5_parser = subparsers.add_parser(
        "phase5-self-paced",
        help="Initialize and run the SPAL-inspired Phase 5 retrospective replay.",
    )
    phase5_subparsers = phase5_parser.add_subparsers(
        dest="phase5_action",
        required=True,
    )

    def add_phase5_common(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--phase1-root", default="thesis_results/01_reproduction")
        parser.add_argument("--phase2-root", default="thesis_results/02_replay")
        parser.add_argument(
            "--output-root",
            default="thesis_results/05_self_paced_active_learning",
        )

    phase5_init_parser = phase5_subparsers.add_parser(
        "init",
        help="Write Phase 5 manifests and separately schedulable SUPEK PBS jobs.",
    )
    add_phase5_common(phase5_init_parser)
    phase5_init_parser.add_argument("--pbs-repo-root", default=None)
    phase5_init_parser.add_argument("--outer-folds", nargs="+", type=int, default=[1, 2, 3])
    phase5_init_parser.add_argument("--inner-fold", type=int, default=1)
    phase5_init_parser.add_argument("--initial-label-counts", nargs="+", type=int, default=[10])
    phase5_init_parser.add_argument("--strategies", nargs="+", choices=PHASE5_STRATEGIES, default=list(PHASE5_STRATEGIES))
    phase5_init_parser.add_argument("--batch-size", type=int, default=5)
    phase5_init_parser.add_argument("--max-rounds", type=int, default=45)
    phase5_init_parser.add_argument("--ensemble-size", type=int, default=1)
    phase5_init_parser.add_argument("--base-seed", type=int, default=20260317)
    phase5_init_parser.add_argument("--supek-walltime", default=None)
    phase5_init_parser.add_argument("--supek-queue", default="gpu")
    phase5_init_parser.add_argument("--supek-ncpus", type=int, default=4)
    phase5_init_parser.add_argument("--supek-ngpus", type=int, default=1)
    phase5_init_parser.add_argument("--supek-mem", default="40GB")
    phase5_init_parser.add_argument("--force", action="store_true")

    phase5_run_parser = phase5_subparsers.add_parser(
        "run-job",
        help="Run one Phase 5 fold, initial-label condition, and strategy trajectory.",
    )
    add_phase5_common(phase5_run_parser)
    phase5_run_parser.add_argument("--outer-fold", type=int, required=True)
    phase5_run_parser.add_argument("--inner-fold", type=int, default=1)
    phase5_run_parser.add_argument("--initial-label-count", type=int, required=True)
    phase5_run_parser.add_argument("--strategy", choices=PHASE5_STRATEGIES, required=True)
    phase5_run_parser.add_argument("--batch-size", type=int, default=5)
    phase5_run_parser.add_argument("--max-rounds", type=int, default=45)
    phase5_run_parser.add_argument("--ensemble-size", type=int, default=1)
    phase5_run_parser.add_argument("--base-seed", type=int, default=20260317)
    phase5_run_parser.add_argument("--force", action="store_true")

    phase5_aggregate_parser = phase5_subparsers.add_parser(
        "aggregate",
        help="Aggregate completed Phase 5 jobs and write thesis tables and figures.",
    )
    add_phase5_common(phase5_aggregate_parser)

    phase5_status_parser = phase5_subparsers.add_parser(
        "status",
        help="Report Phase 5 replay-job completion state.",
    )
    add_phase5_common(phase5_status_parser)

    return parser


def _config_from_args(args: argparse.Namespace) -> RunConfig:
    kwargs = {
        "run_name": args.run_name,
        "output_root": args.output_root,
        "random_seed": args.random_seed,
        "batch_size": args.batch_size,
        "max_rounds": args.max_rounds,
        "epochs": args.epochs,
        "candidate_pool_min": args.candidate_pool_min,
        "replay_seed_size": args.replay_seed_size,
        "real_strategy": args.real_strategy,
        "train_family_for_init": args.train_family_for_init,
        "use_calibrated_acquisition": not args.raw_acquisition,
        "generator_objective_mode": args.generator_objective_mode,
        "use_similarity_penalty": args.use_similarity_penalty,
        "use_length_penalty": not args.no_length_penalty,
        "binary_threshold_strategy": args.binary_threshold_strategy,
    }
    if args.replay_strategies:
        kwargs["replay_strategies"] = args.replay_strategies
    return RunConfig(**kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)


    if args.command == "phase1-reproduce":
        try:
            summary = run_phase1_reproduce(args)
        except ValueError as exc:
            parser.error(str(exc))
            return 2
        if args.status:
            print(format_status_report(summary))
        else:
            print(json.dumps(summary, indent=2))
        return 0

    if args.command == "phase2-replay":
        try:
            summary = run_phase2_replay(args)
        except ValueError as exc:
            parser.error(str(exc))
            return 2
        if args.status:
            print(format_phase2_status(summary))
        else:
            print(json.dumps(summary, indent=2))
        return 0

    if args.command == "phase2-export":
        summary = run_phase2_export(args)
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "phase3-select-strategies":
        summary = run_phase3_strategy_selection(args)
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "phase3-real-al":
        summary = run_phase3_real_al(args)
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "phase4-bo":
        if args.phase4_bo_action == "init-diversity-aware":
            summary = init_phase4_diversity(args)
            print(json.dumps(summary, indent=2))
            return 0
        if args.phase4_bo_action == "run-diversity-aware":
            summary = run_phase4_diversity(args)
            print(json.dumps(summary, indent=2))
            return 0
        summary = run_phase4_bo(args)
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "phase5-self-paced":
        summary = run_phase5(args)
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "init-run":
        run_dir = init_run(_config_from_args(args))
        print(f"Initialized active-learning run at: {run_dir}")
        return 0

    if args.command == "run-replay":
        results = run_replay(Path(args.run_dir), args.strategies)
        print(f"Replay benchmark completed for strategies: {', '.join(results)}")
        return 0

    if args.command == "summarize-study":
        summary = summarize_study(
            Path(args.run_root),
            output_dir=Path(args.output_dir) if args.output_dir else None,
            metric=args.metric,
            target=args.target,
            run_names=args.runs,
        )
        print(
            "Study evidence summarized: "
            f"{summary['strategy_count']} strategies across {summary['run_count']} runs. "
            f"Best by AULC({args.metric})={summary['best_strategy_by_aulc'] or 'n/a'}"
        )
        print(f"Strategy table: {summary['outputs']['strategy_summary']}")
        print(f"Run/strategy table: {summary['outputs']['run_strategy_summary']}")
        print(f"Paired random comparison: {summary['outputs']['paired_vs_random']}")
        return 0

    if args.command == "run-study":
        manifest = run_study(
            study_name=args.study_name,
            run_root=Path(args.run_root),
            seed_count=args.seeds,
            seed_start=args.seed_start,
            seed_step=args.seed_step,
            seed_index_start=args.seed_index_start,
            epochs=args.epochs,
            max_rounds=args.max_rounds,
            batch_size=args.batch_size,
            candidate_pool_min=args.candidate_pool_min,
            replay_seed_size=args.replay_seed_size,
            real_strategy=args.real_strategy,
            replay_strategies=args.strategies,
            ensemble_size=args.ensemble_size,
            train_family_for_init=args.train_family_for_init,
            use_calibrated_acquisition=not args.raw_acquisition,
            generator_objective_mode=args.generator_objective_mode,
            use_similarity_penalty=args.use_similarity_penalty,
            use_length_penalty=not args.no_length_penalty,
            binary_threshold_strategy=args.binary_threshold_strategy,
            metric=args.metric,
            target=args.target,
            summarize=not args.no_summarize,
            dry_run=args.dry_run,
            force_replay=args.force_replay,
            allow_config_mismatch=args.allow_config_mismatch,
        )
        print(
            "Study run finished: "
            f"status={manifest['status']} "
            f"completed={manifest['completed_run_count']}/{manifest['run_count']} "
            f"failures={manifest['failure_count']}"
        )
        print(f"Manifest: {manifest['manifest_path']}")
        outputs = manifest.get("summary", {}).get("outputs", {}) if isinstance(manifest.get("summary"), dict) else {}
        if outputs:
            print(f"Strategy table: {outputs.get('strategy_summary', '')}")
        return 0 if manifest["status"] in {"completed", "planned"} else 1

    if args.command == "compare-studies":
        comparison = compare_studies(
            Path(args.run_root),
            baseline_study=args.baseline_study,
            candidate_study=args.candidate_study,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            metric=args.metric,
            target=args.target,
        )
        print(
            "Study comparison finished: "
            f"{comparison['paired_count']} matched seed/strategy pairs, "
            f"{comparison['strategy_count']} strategies. "
            f"Best by AULC advantage={comparison['best_strategy_by_aulc_advantage'] or 'n/a'}"
        )
        print(f"Strategy delta table: {comparison['outputs']['strategy_summary']}")
        print(f"Paired delta table: {comparison['outputs']['paired_comparison']}")
        print(f"Thesis narrative: {comparison['outputs']['thesis_narrative']}")
        return 0 if comparison["paired_count"] else 1

    if args.command == "propose-round":
        batch_path = propose_round(Path(args.run_dir), args.strategy)
        print(f"Exported CG-MD batch to: {batch_path}")
        return 0

    if args.command == "ingest-round":
        metrics = ingest_round(Path(args.run_dir), Path(args.import_csv))
        print(
            "Imported CG-MD labels and updated validation metrics: "
            f"F1={metrics['f1']:.4f}, PR_AUC={metrics['pr_auc']:.4f}, "
            f"ROC_AUC={metrics['roc_auc']:.4f}"
        )
        return 0

    if args.command == "run-discovery":
        results = run_discovery(Path(args.run_dir), args.strategies)
        print(f"Discovery completed for strategies: {', '.join(results)}")
        return 0

    if args.command == "evaluate-final":
        metrics = evaluate_final(Path(args.run_dir))
        print(
            "Final holdout evaluation completed: "
            f"F1={metrics['f1']:.4f}, PR_AUC={metrics['pr_auc']:.4f}, "
            f"ROC_AUC={metrics['roc_auc']:.4f}"
        )
        return 0

    if args.command == "freeze-final":
        report = freeze_final_result(
            Path(args.run_dir),
            run_evaluation=args.run_evaluation,
            force=args.force,
            allow_unresolved=args.allow_unresolved,
            metric=args.metric,
        )
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            counts = report.get("counts", {}) if isinstance(report.get("counts", {}), dict) else {}
            outputs = report.get("outputs", {}) if isinstance(report.get("outputs", {}), dict) else {}
            print(
                "Final thesis freeze completed: "
                f"status={report.get('status', '')} "
                f"blockers={counts.get('failed_blockers', 0)} "
                f"warnings={counts.get('failed_warnings', 0)}"
            )
            print(f"Freeze manifest: {outputs.get('freeze_json', '')}")
            print(f"Model card: {outputs.get('model_card', '')}")
        if report.get("status") == "blocked":
            return 1
        return 0

    if args.command == "prepare-md-campaign":
        campaign_dir = prepare_md_campaign(
            Path(args.run_dir),
            Path(args.batch_csv),
            args.campaign,
            args.cluster,
            args.md_profile,
        )
        print(f"Prepared MD campaign at: {campaign_dir}")
        return 0

    if args.command == "build-pdbs":
        pdb_dir = build_pdbs(Path(args.campaign_dir), validate_only=args.validate_only)
        print(f"PDB staging ready at: {pdb_dir}")
        return 0

    if args.command == "parse-md-results":
        review_path = parse_md_results(Path(args.campaign_dir))
        print(f"Parsed MD results into: {review_path}")
        return 0

    if args.command == "make-md-ingest-csv":
        ingest_path = make_md_ingest_csv(Path(args.campaign_dir), Path(args.review_csv))
        print(f"Generated ingest CSV at: {ingest_path}")
        return 0

    if args.command == "prepare-md-stage":
        campaign_dir, next_commands_path = prepare_md_stage(
            Path(args.run_dir),
            Path(args.batch_csv),
            args.sequence,
            args.campaign,
            args.md_profile,
            cluster=args.cluster,
            reuse_pdb_from=Path(args.reuse_pdb_from) if args.reuse_pdb_from else None,
            exclude_nodes=args.exclude_nodes,
        )
        print(f"Prepared guided MD stage at: {campaign_dir}")
        print(f"Next BURA commands written to: {next_commands_path}")
        print(next_commands_path.read_text(encoding="utf-8").rstrip())
        return 0

    if args.command == "prepare-manual-md-stage":
        campaign_dir, next_commands_path, batch_csv = prepare_manual_md_stage(
            Path(args.run_dir),
            args.sequence,
            args.campaign,
            args.md_profile,
            cluster=args.cluster,
            reuse_pdb_from=Path(args.reuse_pdb_from) if args.reuse_pdb_from else None,
            exclude_nodes=args.exclude_nodes,
        )
        print(f"Prepared manual MD sandbox stage at: {campaign_dir}")
        print(f"Manual source batch written to: {batch_csv}")
        print(f"Next BURA commands written to: {next_commands_path}")
        print(next_commands_path.read_text(encoding="utf-8").rstrip())
        return 0

    if args.command == "finalize-md-stage":
        review_path, review_row, next_message = finalize_md_stage(
            Path(args.campaign_dir),
            Path(args.staged_package_dir) if args.staged_package_dir else None,
        )
        print(f"Parsed MD results into: {review_path}")
        print(
            f"Sequence={review_row['sequence']} "
            f"md_profile={review_row['md_profile']} "
            f"job_root_status={review_row['job_root_status']}"
        )
        print(next_message)
        return 0

    if args.command == "bura-full-autopilot":
        summary = run_bura_full_autopilot(
            run_root=Path(args.run_root),
            campaign_dir=Path(args.campaign_dir),
            sequence=args.sequence,
            exclude_nodes=args.exclude_nodes,
            poll_seconds=args.poll_seconds,
            max_wait_seconds=args.max_wait_seconds,
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "bura-recover-md-outputs":
        summary = recover_bura_outputs_only(
            run_root=Path(args.run_root),
            campaign_dir=Path(args.campaign_dir),
            sequence=args.sequence,
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "dashboard-run-bura-md-workflow":
        summary = run_bura_md_workflow_from_payload(args.payload_json)
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "prepare-bura-md-benchmark":
        benchmark_dir = prepare_bura_md_benchmark(
            Path(args.campaign_dir),
            args.sequence,
            benchmark_name=args.benchmark_name,
            nsteps=args.nsteps,
            layouts=args.layouts or None,
            walltime=args.walltime,
        )
        next_commands_path = benchmark_dir / "NEXT_BURA_BENCHMARK_COMMANDS.md"
        print(f"Prepared BURA MD benchmark at: {benchmark_dir}")
        print(f"Next BURA commands written to: {next_commands_path}")
        print(next_commands_path.read_text(encoding="utf-8").rstrip())
        return 0

    if args.command == "parse-bura-md-benchmark":
        results_path = parse_bura_md_benchmark(Path(args.benchmark_dir))
        print(f"Parsed BURA MD benchmark into: {results_path}")
        return 0

    if args.command == "md-ladder-status":
        status = md_ladder_status(Path(args.run_dir), args.sequence)
        print(f"Sequence: {status['sequence']}")
        if status['campaigns']:
            for campaign in status['campaigns']:
                print(
                    f"- {campaign['campaign']} "
                    f"({campaign['md_profile']}): {campaign['job_root_status']}"
                )
        else:
            print("- No guided MD campaigns found for this sequence.")
        if status['next_profile']:
            print(f"Next recommended profile: {status['next_profile']}")
        if status['ready_for_review']:
            print("Ready for review and ingest prep: yes")
        else:
            print("Ready for review and ingest prep: no")
        return 0

    if args.command == "dashboard":
        return launch_dashboard(
            Path(args.run_root),
            host=args.host,
            port=args.port,
            refresh_seconds=args.refresh_seconds,
            skip_integrity_check=args.skip_integrity_check,
        )

    if args.command == "dashboard-check":
        dashboard_args = ["--run-root", args.run_root, "--refresh-seconds", str(args.refresh_seconds)]
        if args.json:
            dashboard_args.append("--json")
        return dashboard_integrity_main(dashboard_args)

    if args.command == "dashboard-smoke":
        dashboard_args = ["--run-root", args.run_root, "--refresh-seconds", str(args.refresh_seconds)]
        if args.json:
            dashboard_args.append("--json")
        if args.tests:
            dashboard_args.extend(["--tests", *args.tests])
        return dashboard_smoke_main(dashboard_args)

    if args.command == "thesis-canary":
        report = run_thesis_canary(
            run_root=Path(args.run_root),
            name=args.name,
            seed=args.seed,
            peptide_count=args.peptides,
            force=args.force,
        )
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            checks = report.get("checks", {}) if isinstance(report.get("checks", {}), dict) else {}
            outputs = report.get("outputs", {}) if isinstance(report.get("outputs", {}), dict) else {}
            print(
                "Thesis canary finished: "
                f"status={report['status']} seed={report['seed']} peptides={report['peptide_count']}"
            )
            print(
                "Checks: "
                f"analysis_complete={checks.get('analysis_complete_rows', 0)} "
                f"evidence_backed={checks.get('evidence_backed_reviews', 0)} "
                f"ingest_validated={checks.get('ingest_rows_validated', 0)}"
            )
            print(f"Report: {outputs.get('report_markdown', '')}")
            print(f"Run dir: {report.get('run_dir', '')}")
        return 0 if report.get("status") == "passed" else 1

    if args.command == "export-thesis-packet":
        packet = export_thesis_packet(
            Path(args.run_root),
            output_dir=Path(args.output_dir) if args.output_dir else None,
            title=args.title,
            metric=args.metric,
            include_dashboard=not args.skip_dashboard,
        )
        if args.json:
            print(json.dumps(packet, indent=2))
        else:
            counts = packet.get("counts", {}) if isinstance(packet.get("counts", {}), dict) else {}
            outputs = packet.get("outputs", {}) if isinstance(packet.get("outputs", {}), dict) else {}
            print(
                "Thesis packet exported: "
                f"runs={counts.get('runs', 0)} "
                f"md_reviews={counts.get('md_review_rows', 0)} "
                f"canaries={counts.get('canaries', 0)}"
            )
            print(f"Packet index: {outputs.get('index', '')}")
            print(f"Manifest: {outputs.get('manifest', '')}")
        return 0

    if args.command == "build-thesis-figures":
        bundle = build_thesis_figures(
            Path(args.packet_dir),
            output_dir=Path(args.output_dir) if args.output_dir else None,
            metric=args.metric,
        )
        if args.json:
            print(json.dumps(bundle, indent=2))
        else:
            counts = bundle.get("counts", {}) if isinstance(bundle.get("counts", {}), dict) else {}
            outputs = bundle.get("outputs", {}) if isinstance(bundle.get("outputs", {}), dict) else {}
            print(
                "Thesis figures built: "
                f"figures={counts.get('figures', 0)} "
                f"skipped={counts.get('skipped_figures', 0)} "
                f"captions={counts.get('captions', 0)}"
            )
            print(f"Manifest: {outputs.get('manifest', '')}")
            print(f"Captions: {outputs.get('figure_captions', '')}")
        return 0

    if args.command == "optional-evaluator-study":
        report = run_optional_evaluator_study(
            Path(args.run_dir),
            external_scores=Path(args.external_scores) if args.external_scores else None,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            bin_count=args.bin_count,
        )
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            counts = report.get("counts", {}) if isinstance(report.get("counts", {}), dict) else {}
            outputs = report.get("outputs", {}) if isinstance(report.get("outputs", {}), dict) else {}
            print(
                "Optional evaluator study exported: "
                f"status={report.get('status', '')} "
                f"internal={counts.get('unique_internal_sequences', 0)} "
                f"external={counts.get('external_rows', 0)} "
                f"matched={counts.get('disagreement_rows', 0)}"
            )
            print(f"Manifest: {outputs.get('manifest', '')}")
            print(f"README: {outputs.get('readme', '')}")
        return 0

    if args.command == "dashboard-run-md-slate":
        run_md_slate_supervisor(Path(args.run_root), args.slate_id)
        return 0

    if args.command == "dashboard-continue-feedback":
        summary = run_feedback_loop(Path(args.run_dir), propose_next_batch=args.propose_next_batch)
        print(
            "Closed the MD feedback loop for round "
            f"{summary['pending_round_id']} with sequences: {', '.join(summary['pending_sequences'])}"
        )
        print(f"Aggregate import CSV: {summary['aggregate_import_csv']}")
        print(f"Synced per-peptide ingest CSVs: {len(summary['synced_ingest_csvs'])}")
        metrics = summary.get("metrics", {})
        if isinstance(metrics, dict):
            print(
                "Updated validation metrics: "
                f"F1={metrics.get('f1', 0.0):.4f}, PR_AUC={metrics.get('pr_auc', 0.0):.4f}, "
                f"ROC_AUC={metrics.get('roc_auc', 0.0):.4f}"
            )
        if summary.get("next_batch_csv"):
            print(f"Proposed next batch: {summary['next_batch_csv']}")
        return 0

    if args.command == "dashboard-init-run":
        summary = dashboard_init_run(
            run_root=Path(args.run_root),
            run_name=args.run_name,
            random_seed=args.random_seed,
            batch_size=args.batch_size,
            max_rounds=args.max_rounds,
            epochs=args.epochs,
            candidate_pool_min=args.candidate_pool_min,
            replay_seed_size=args.replay_seed_size,
            real_strategy=args.real_strategy,
            replay_strategies=parse_strategy_list(args.replay_strategies),
            train_family_for_init=args.train_family_for_init,
            use_calibrated_acquisition=not args.raw_acquisition,
            generator_objective_mode=args.generator_objective_mode,
            use_similarity_penalty=args.use_similarity_penalty,
            use_length_penalty=not args.no_length_penalty,
            binary_threshold_strategy=args.binary_threshold_strategy,
            pin_run=args.pin_run,
            run_label=args.run_label,
            train_baseline_after_init=not args.skip_baseline_init,
            run_replay_after_init=args.run_replay_after_init,
        )
        print(f"Initialized dashboard run: {summary['run_dir']}")
        if summary.get("pinned"):
            print("Pinned for Current Thesis Work.")
        if summary.get("label"):
            print(f"Run label: {summary['label']}")
        if not summary.get("baseline_trained"):
            print("Baseline training skipped; run is config-only until a local or remote workflow trains it.")
        if summary.get("replay_started"):
            print("Replay completed for strategies: " + ", ".join(summary.get("replay_strategies", [])))
        return 0

    parser.error("Unknown command")
    return 2
