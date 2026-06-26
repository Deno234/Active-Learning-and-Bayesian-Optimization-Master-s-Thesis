from __future__ import annotations

from pathlib import Path
from types import ModuleType

from active_learning_thesis.config import (
    BINARY_THRESHOLD_STRATEGIES,
    GENERATOR_OBJECTIVE_MODES,
    RunConfig,
)


OPERATIONS_SECTIONS = [
    "Cluster health",
    "Notifications",
    "Action debugger",
    "Thesis checklist",
    "Runbook",
    "New run wizard",
    "Study designer",
    "Execution readiness",
    "Action contracts",
    "AL loop simulator",
    "Approval queue",
    "Remote jobs",
    "Recovery center",
    "Transfers",
    "Run curation",
]
GUIDED_OPERATIONS_SECTIONS = [
    "Cluster health",
    "Notifications",
    "Thesis checklist",
    "New run wizard",
    "Study designer",
    "Remote jobs",
    "Advanced / debug tools",
]
ADVANCED_OPERATIONS_SECTIONS = [
    section
    for section in OPERATIONS_SECTIONS
    if section not in GUIDED_OPERATIONS_SECTIONS and section != "Advanced / debug tools"
]


RUNBOOK_GUIDES = [
    {
        "topic": "Debug a failed GUI action",
        "safe_first_move": "Open Operations -> Action debugger and read the one-line verdict before retrying anything.",
        "where": "Operations -> Action debugger, then the linked workflow page if the safe next move names one.",
        "do_next": "Copy the debug packet if you need to preserve the failure for thesis notes or ask for help later.",
        "do_not": "Do not immediately rerun failed remote or ingest actions before the debugger points to a read-only check or artifact fix.",
    },
    {
        "topic": "Start a new thesis run",
        "safe_first_move": "Open Operations -> New run wizard and choose a preset before editing individual numbers.",
        "where": "Operations -> New run wizard.",
        "do_next": "Use Quick smoke for a tiny check, Thesis baseline for normal work, or Clone existing run when you want the same setup with a new seed/name.",
        "do_not": "Do not reuse an existing run folder name; the wizard blocks that to protect previous evidence.",
    },
    {
        "topic": "Design a multi-seed study",
        "safe_first_move": "Open Operations -> Study designer and run a dry-run plan before spending real compute.",
        "where": "Operations -> Study designer, then Results -> Study comparison hub.",
        "do_next": "Use Strategy comparison for normal thesis evidence, then summarize or compare completed studies from the same page.",
        "do_not": "Do not compare studies with different seeds/strategies unless the comparison summary shows matched pairs.",
    },
    {
        "topic": "Daily thesis cockpit check",
        "safe_first_move": "Open Today, read the top recommendation, then check unread notifications before launching new work.",
        "where": "Today -> Notifications, then Operations -> Notifications if anything needs acknowledging.",
        "do_next": "If the top item is remote/MD related, open MD Validation or Operations before touching Model Workflow.",
        "do_not": "Do not launch new BURA work while artifact verification or slate recovery is already asking for attention.",
    },
    {
        "topic": "Choose and launch the next MD slate",
        "safe_first_move": "Open Peptides -> Candidate selection and filter to the shortlist that matches the thesis question you want to answer.",
        "where": "Peptides -> Candidate selection -> Selected MD slate.",
        "do_next": "Mark candidates Selected for MD, save a named slate plan with rationale, inspect blockers/cap fit, then launch the saved plan.",
        "do_not": "Do not quick-launch discovery-only peptides until the GUI shows a source batch row or creates a dashboard-local MD batch for reporting-only validation.",
    },
    {
        "topic": "Monitor an active MD slate",
        "safe_first_move": "Open MD Validation -> Slate monitor and inspect the shared slate board before using single-peptide controls.",
        "where": "MD Validation -> Slate monitor, or Operations -> Remote jobs for the global view.",
        "do_next": "Use stage/status/job-id rows to decide whether to wait, fetch logs, recover a stale peptide, or pause the slate.",
        "do_not": "Do not manually rerun a stage from outside the slate unless the Recovery center says the tracked child action is lost or stale.",
    },
    {
        "topic": "Rehearse the full AL loop safely",
        "safe_first_move": "Open Operations -> AL loop simulator before a high-risk candidate-to-retrain handoff.",
        "where": "Operations -> AL loop simulator.",
        "do_next": "Start a local rehearsal, inject simulated labels, simulate ingest, then simulate retrain/propose to inspect the exact real-path files and commands.",
        "do_not": "Do not treat simulated labels as real `cgmd_label` decisions; the simulator never replaces your manual post-analysis call.",
    },
    {
        "topic": "Sanity-check a risky action before clicking it",
        "safe_first_move": "Open Operations -> Action contracts when you want to know exactly what a high-risk button will write or require.",
        "where": "Operations -> Action contracts, or the per-button Action contract expander on the page itself.",
        "do_next": "Read the prerequisites, side effects, safer option, and recovery path before launching remote work or closing the AL loop.",
        "do_not": "Do not assume that two similar buttons do the same thing; use the contract when the difference matters.",
    },
    {
        "topic": "Recover a blocked or stale MD peptide",
        "safe_first_move": "Open the Recovery center and read the specific reason before retrying anything.",
        "where": "MD Validation -> Recovery center for the run, or Operations -> Recovery center across all visible work.",
        "do_next": "Prefer Rebind latest tracked action if the job still exists; use Recover from last checkpoint when the action is gone or stale.",
        "do_not": "Do not cancel or relaunch a remote job just because a peptide is waiting on a cap or poll window.",
    },
    {
        "topic": "Handle BURA failure or queue block",
        "safe_first_move": "Open Operations -> Remote jobs and read the BURA live console plus scheduler/hold history.",
        "where": "Operations -> Remote jobs, then MD Validation -> Recovery center if a slate peptide is affected.",
        "do_next": "Run readiness/preflight again before resubmitting; if logs show missing files, fix artifact/source-batch issues first.",
        "do_not": "Do not retry a full chain before the latest readiness result and artifact checks agree the package is complete.",
    },
    {
        "topic": "Handle SUPEK model-workflow failure",
        "safe_first_move": "Open Model Workflow -> Remote SUPEK for the selected run and inspect the latest console snapshot.",
        "where": "Model Workflow -> Remote SUPEK, or Operations -> Remote jobs for global logs.",
        "do_next": "Check SUPEK health and submit preflight, then fetch logs before rerunning sync/submit actions.",
        "do_not": "Do not assume a remote model job finished just because it disappeared from the queue; fetch logs or pull artifacts first.",
    },
    {
        "topic": "Fix artifact verification problems",
        "safe_first_move": "Open the artifact row and identify whether the missing file is staged, copied-back, analysis, or ingest-related.",
        "where": "MD Validation -> Artifact verification or Operations -> Transfers.",
        "do_next": "For remote output gaps, re-pull or finalize the package. For source-batch gaps, create/promote the correct batch row before ingest.",
        "do_not": "Do not create `cgmd_ingest.csv` from an incomplete or reporting-only campaign unless the promotion bridge has made it a real AL batch.",
    },
    {
        "topic": "Review MD outputs and assign labels",
        "safe_first_move": "Open MD Validation -> Review & ingest and inspect the returned full-analysis evidence before editing the label.",
        "where": "MD Validation -> Review & ingest, with Peptides -> Bulk review / ingest for multi-row review.",
        "do_next": "Write the human `cgmd_label` and notes, then create ingest CSV only when the campaign is ingest-supported.",
        "do_not": "Do not let SASA/AP outputs decide the label automatically; they are evidence, while the final self-assembly call is yours.",
    },
    {
        "topic": "Feed reviewed labels back into AL",
        "safe_first_move": "Open Model Workflow and check the feedback queue before running any ingest action.",
        "where": "Model Workflow -> Local model actions, or Today when Continue AL from reviewed peptides is shown.",
        "do_next": "Run Continue AL from reviewed peptides when all pending batch rows are reviewed and ingestable, then optionally propose the next round.",
        "do_not": "Do not ingest a partial pending batch unless the queue explicitly says that is the intended thesis step.",
    },
    {
        "topic": "Prepare thesis evidence/export",
        "safe_first_move": "Open Results and decide whether you want selected runs only or report-ready-only evidence.",
        "where": "Results -> Appendix packet, Results -> Peptide provenance audit, Results -> Promotion audit.",
        "do_next": "Use the export blocks for scorecards, provenance, promotion, decisions, and figure captions.",
        "do_not": "Do not copy historical/test runs into thesis notes unless you intentionally selected All Runs or Historical / Test.",
    },
]


def _runbook_rows() -> list[dict[str, str]]:
    return [
        {
            "topic": str(guide["topic"]),
            "safe_first_move": str(guide["safe_first_move"]),
            "where": str(guide["where"]),
            "do_next": str(guide["do_next"]),
            "do_not": str(guide["do_not"]),
        }
        for guide in RUNBOOK_GUIDES
    ]


def _remote_watchdog_display_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "verdict": row.get("verdict", ""),
            "cluster": str(row.get("cluster", "")).upper(),
            "target": row.get("sequence", "") or row.get("run", "") or row.get("campaign", ""),
            "source": row.get("source", ""),
            "state": row.get("dashboard_state", ""),
            "job_id": row.get("remote_job_id", "") or "-",
            "queue": row.get("queue_state", ""),
            "snapshot_age_h": row.get("snapshot_age_hours", ""),
            "health": row.get("health_status", ""),
            "reason": row.get("reason", ""),
            "safe_next_move": row.get("safe_next_move", ""),
            "read_only_followup": row.get("recommended_action", ""),
            "open_view": row.get("open_view", ""),
        }
        for row in rows
    ]


def _form_container(st, *, key: str):
    form = getattr(st, "form", None)
    if callable(form):
        try:
            return form(key=key, clear_on_submit=False, border=False)
        except TypeError:
            return form(key=key, clear_on_submit=False)
    return st.container()


def _form_submit_button(st, label: str, *, key: str, disabled: bool = False) -> bool:
    form_submit_button = getattr(st, "form_submit_button", None)
    if callable(form_submit_button):
        return bool(form_submit_button(label, disabled=disabled))
    return bool(st.button(label, key=key, disabled=disabled))


def _render_study_run_ablation_form(
    st,
    *,
    state: dict[str, object],
    run_root: Path,
    study_rows: list[dict[str, object]],
    STUDY_PRESETS: dict[str, dict[str, object]],
    METRIC_FIELDS: list[str],
    coerce_positive_int,
    normalize_study_name,
    parse_float_or_none,
    parse_strategy_list,
    _friendly_bool,
    _quote_path,
    get_cluster_profile,
    build_button_readiness_contract,
    submit_run_study_action,
    draft_supek_submit_study_action,
    draft_supek_submit_study_array_action,
    list_dashboard_actions,
    _render_action_history,
    _report_action_result,
    APPROVAL_PENDING_STATUSES,
) -> None:
    preset_name = st.selectbox(
        "Study preset",
        list(STUDY_PRESETS),
        index=0,
        key="study_designer_preset",
    )
    preset = dict(STUDY_PRESETS[preset_name])
    st.info(str(preset.get("description", "")))
    default_study_name = normalize_study_name(f"{preset.get('study_name_prefix', 'study')}_{len(study_rows) + 1:03d}")
    raw_study_name = st.text_input("Study name", value=default_study_name, key="study_designer_name")
    study_name = normalize_study_name(raw_study_name, fallback="")
    if study_name != str(raw_study_name).strip():
        st.caption(f"Normalized study name: `{study_name}`")
    metric = st.selectbox(
        "Primary metric",
        METRIC_FIELDS,
        index=METRIC_FIELDS.index(str(preset.get("metric", "f1"))) if str(preset.get("metric", "f1")) in METRIC_FIELDS else 0,
        key="study_designer_metric",
    )
    target_text = st.text_input("Optional target metric", value=str(preset.get("target", "")), key="study_designer_target")
    seeds = coerce_positive_int(st.text_input("Seed count", value=str(preset.get("seeds", 5)), key="study_designer_seeds"), int(preset.get("seeds", 5)))
    seed_start = coerce_positive_int(st.text_input("Seed start", value=str(preset.get("seed_start", 20260317)), key="study_designer_seed_start"), int(preset.get("seed_start", 20260317)), minimum=0)
    seed_step = coerce_positive_int(st.text_input("Seed step", value=str(preset.get("seed_step", 1009)), key="study_designer_seed_step"), int(preset.get("seed_step", 1009)))
    epochs = coerce_positive_int(st.text_input("Epochs", value=str(preset.get("epochs", 70)), key="study_designer_epochs"), int(preset.get("epochs", 70)))
    max_rounds = coerce_positive_int(st.text_input("Max rounds", value=str(preset.get("max_rounds", 10)), key="study_designer_max_rounds"), int(preset.get("max_rounds", 10)), minimum=0)
    batch_size = coerce_positive_int(st.text_input("Batch size", value=str(preset.get("batch_size", 5)), key="study_designer_batch_size"), int(preset.get("batch_size", 5)))
    candidate_pool_min = coerce_positive_int(st.text_input("Candidate pool minimum", value=str(preset.get("candidate_pool_min", 50)), key="study_designer_candidate_pool"), int(preset.get("candidate_pool_min", 50)))
    replay_seed_size = coerce_positive_int(st.text_input("Replay seed size", value=str(preset.get("replay_seed_size", 40)), key="study_designer_replay_seed"), int(preset.get("replay_seed_size", 40)))
    ensemble_size = coerce_positive_int(st.text_input("Ensemble size", value=str(preset.get("ensemble_size", 5)), key="study_designer_ensemble_size"), int(preset.get("ensemble_size", 5)))
    strategy_text = st.text_input(
        "Replay strategies",
        value=", ".join(str(item) for item in list(preset.get("strategies", []))),
        key="study_designer_strategies",
    )
    strategies = parse_strategy_list(strategy_text)
    real_strategy_options = list(dict.fromkeys([str(preset.get("real_strategy", "ensemble_mi")), *strategies, "ensemble_mi", "family_qbc"]))
    real_strategy = st.selectbox(
        "Real AL strategy",
        real_strategy_options,
        index=0,
        key="study_designer_real_strategy",
    )
    train_family_for_init = st.checkbox("Train family committee during init", value=bool(preset.get("train_family_for_init", False)), key="study_designer_train_family")
    use_calibrated_acquisition = st.checkbox("Use calibrated acquisition", value=bool(preset.get("use_calibrated_acquisition", True)), key="study_designer_calibrated")
    generator_objective_mode = st.selectbox(
        "Generator objective mode",
        GENERATOR_OBJECTIVE_MODES,
        index=GENERATOR_OBJECTIVE_MODES.index(
            str(preset.get("generator_objective_mode", RunConfig().generator_objective_mode))
            if str(preset.get("generator_objective_mode", RunConfig().generator_objective_mode)) in GENERATOR_OBJECTIVE_MODES
            else RunConfig().generator_objective_mode
        ),
        key="study_designer_generator_objective_mode",
        help="Stored in each run config. Replay remains a hidden-pool acquisition comparison; real AL proposal uses this.",
    )
    use_similarity_penalty = st.checkbox(
        "Use similarity penalty during generation",
        value=bool(preset.get("use_similarity_penalty", False)),
        key="study_designer_similarity_penalty",
    )
    use_length_penalty = st.checkbox(
        "Use length penalty during generation",
        value=bool(preset.get("use_length_penalty", True)),
        key="study_designer_length_penalty",
    )
    threshold_default = str(
        preset.get(
            "binary_threshold_strategy",
            RunConfig().binary_threshold_strategy,
        )
    )
    if threshold_default not in BINARY_THRESHOLD_STRATEGIES:
        threshold_default = RunConfig().binary_threshold_strategy
    binary_threshold_strategy = st.selectbox(
        "Binary threshold strategy",
        BINARY_THRESHOLD_STRATEGIES,
        index=BINARY_THRESHOLD_STRATEGIES.index(threshold_default),
        key="study_designer_binary_threshold_strategy",
        help="New-run primary F1 is PR-best-F1 thresholded F1; fixed-0.5 metrics are still reported as secondary columns.",
    )
    run_on_supek = st.checkbox("Submit real study to SUPEK instead of local dashboard queue", value=False, key="study_designer_run_on_supek")
    split_supek_by_seed = st.checkbox(
        "Split SUPEK study into one job per seed plus aggregate job",
        value=int(seeds) > 1,
        key="study_designer_split_supek_by_seed",
    )
    supek_profile = get_cluster_profile(state.get("profiles", {}), "supek") if run_on_supek else None
    default_walltime = str((supek_profile or {}).get("default_walltime", "12:00:00") or "12:00:00")
    supek_walltime = st.text_input(
        "SUPEK walltime",
        value=default_walltime if default_walltime != "01:00:00" else "12:00:00",
        key="study_designer_supek_walltime",
    )
    aggregate_walltime = st.text_input(
        "SUPEK aggregate walltime",
        value="02:00:00",
        key="study_designer_supek_aggregate_walltime",
        help="Small dependency job after all seed jobs finish. It rebuilds the full study manifest and summary.",
    )
    dry_run = st.checkbox("Dry run only: write study plan without training", value=True, key="study_designer_dry_run")
    force_replay = st.checkbox("Force replay even if summaries exist", value=False, key="study_designer_force_replay")
    summarize = st.checkbox("Summarize automatically after study completes", value=True, key="study_designer_summarize")
    allow_config_mismatch = st.checkbox("Allow config mismatch when resuming", value=False, key="study_designer_allow_mismatch")
    target = parse_float_or_none(target_text)
    target_text_invalid = bool(str(target_text).strip()) and target is None
    if target_text_invalid:
        st.warning("Target must be numeric before the study can be queued.")
    if run_on_supek and supek_profile is None:
        st.warning("SUPEK profile is not configured yet, so remote study submission is blocked.")

    preview_rows = [
        {"setting": "Study name", "value": study_name},
        {"setting": "Run root", "value": str(run_root)},
        {"setting": "Seeds", "value": seeds},
        {"setting": "Seed start / step", "value": f"{seed_start} / {seed_step}"},
        {"setting": "Metric / target", "value": f"{metric} / {target if target is not None else '-'}"},
        {"setting": "Strategies", "value": ", ".join(strategies)},
        {"setting": "Generator objective mode", "value": generator_objective_mode},
        {"setting": "Similarity / length penalty", "value": f"{_friendly_bool(use_similarity_penalty)} / {_friendly_bool(use_length_penalty)}"},
        {"setting": "Binary threshold strategy", "value": binary_threshold_strategy},
        {"setting": "Expected run folders", "value": seeds},
        {"setting": "Dry run", "value": _friendly_bool(dry_run)},
        {"setting": "Execution target", "value": f"SUPEK ({supek_walltime or '-'})" if run_on_supek else "Local dashboard queue"},
        {"setting": "SUPEK split mode", "value": "per-seed + aggregate" if run_on_supek and split_supek_by_seed else "-"},
        {"setting": "Auto-summarize", "value": _friendly_bool(summarize)},
    ]
    st.markdown("#### Study plan preview")
    st.dataframe(preview_rows)

    study_command = (
        "python -m active_learning_thesis run-study "
        f"--study-name {study_name} "
        f"--run-root {_quote_path(run_root)} "
        f"--seeds {seeds} --seed-start {seed_start} --seed-step {seed_step} "
        f"--epochs {epochs} --max-rounds {max_rounds} --batch-size {batch_size} "
        f"--candidate-pool-min {candidate_pool_min} --replay-seed-size {replay_seed_size} "
        f"--real-strategy {real_strategy} --ensemble-size {ensemble_size} --metric {metric} "
        f"--generator-objective-mode {generator_objective_mode} "
        f"--binary-threshold-strategy {binary_threshold_strategy}"
    )
    if strategies:
        study_command += " --strategies " + " ".join(strategies)
    if target is not None:
        study_command += f" --target {target}"
    if train_family_for_init:
        study_command += " --train-family-for-init"
    if not use_calibrated_acquisition:
        study_command += " --raw-acquisition"
    if use_similarity_penalty:
        study_command += " --use-similarity-penalty"
    if not use_length_penalty:
        study_command += " --no-length-penalty"
    if dry_run:
        study_command += " --dry-run"
    if force_replay:
        study_command += " --force-replay"
    if not summarize:
        study_command += " --no-summarize"
    if allow_config_mismatch:
        study_command += " --allow-config-mismatch"

    study_blockers = []
    study_cautions = []
    if not study_name:
        study_blockers.append("Study name is empty.")
    if not strategies:
        study_blockers.append("At least one replay strategy is required.")
    if target_text_invalid:
        study_blockers.append("Target metric must be a finite number or blank.")
    if run_on_supek and supek_profile is None:
        study_blockers.append("SUPEK profile is not configured.")
    if run_on_supek and dry_run:
        study_cautions.append("Dry-run mode is enabled; the SUPEK job will only write a remote study plan and will not train.")
    if run_on_supek and split_supek_by_seed:
        study_cautions.append("Split mode submits one independent seed job per seed plus one dependency aggregate job. This is faster and safer against timeout, but uses more queue slots.")
    if study_name and (run_root / "_studies" / study_name).exists():
        study_cautions.append("Study directory already exists; this action will resume or update the existing manifest.")
    if not run_root.exists():
        study_cautions.append(f"Run root will be created: {run_root}")
    study_readiness = {
        "verdict": "Blocked" if study_blockers else ("Ready with caution" if study_cautions else "Ready"),
        "summary": "Review the study plan before queueing it.",
        "blockers": study_blockers,
        "cautions": study_cautions,
        "fix_now": "Use a non-empty study name, at least one replay strategy, and a numeric target if provided.",
        "disable_button": bool(study_blockers),
    }
    button_disabled = bool(study_readiness.get("disable_button", False))

    st.markdown("#### Submit")
    if run_on_supek:
        submitter = draft_supek_submit_study_array_action if split_supek_by_seed else draft_supek_submit_study_action
        button_text = "Draft split SUPEK study jobs" if split_supek_by_seed else "Draft SUPEK study job"
        contract_id = "supek-submit-study-array" if split_supek_by_seed else "supek-submit-study"
        action_label = "Submit split study on SUPEK" if split_supek_by_seed else "Submit study on SUPEK"
        command_for_display = study_command.replace(
            f"--run-root {_quote_path(run_root)}",
            f"--run-root {str((supek_profile or {}).get('scratch_run_root', '<supek scratch>'))}",
        )
        build_button_readiness_contract(st, contract_id, readiness=study_readiness)
    else:
        submitter = submit_run_study_action
        button_text = "Queue study"
        contract_id = "run-study"
        action_label = "Run study / ablation"
        command_for_display = study_command
        build_button_readiness_contract(st, contract_id, readiness=study_readiness)
    st.write(action_label)
    if button_disabled:
        st.info("This action is currently blocked. Clear the blocker above and the submit button will re-enable.")
    st.code(command_for_display, language="bash")
    confirm_key = "confirm_study_designer_submit_supek_study" if run_on_supek else "confirm_study_designer_run_study"
    run_key = "run_study_designer_submit_supek_study" if run_on_supek else "run_study_designer_run_study"
    confirmed = st.checkbox(f"Confirm {action_label.lower()}", key=confirm_key)
    submitted = _form_submit_button(st, button_text, key=run_key, disabled=button_disabled)
    if submitted:
        if not confirmed:
            st.warning("Confirm the command first, then submit it.")
            return
        try:
            if run_on_supek:
                action = submitter(
                    run_root=run_root,
                    study_name=study_name,
                    profile=supek_profile,
                    seeds=int(seeds),
                    seed_start=int(seed_start),
                    seed_step=int(seed_step),
                    epochs=int(epochs),
                    max_rounds=int(max_rounds),
                    batch_size=int(batch_size),
                    candidate_pool_min=int(candidate_pool_min),
                    replay_seed_size=int(replay_seed_size),
                    real_strategy=str(real_strategy),
                    strategies=list(strategies),
                    metric=str(metric),
                    target=target,
                    ensemble_size=int(ensemble_size),
                    train_family_for_init=bool(train_family_for_init),
                    use_calibrated_acquisition=bool(use_calibrated_acquisition),
                    generator_objective_mode=str(generator_objective_mode),
                    use_similarity_penalty=bool(use_similarity_penalty),
                    use_length_penalty=bool(use_length_penalty),
                    binary_threshold_strategy=str(binary_threshold_strategy),
                    dry_run=bool(dry_run),
                    force_replay=bool(force_replay),
                    summarize=bool(summarize),
                    allow_config_mismatch=bool(allow_config_mismatch),
                    walltime=str(supek_walltime),
                    aggregate_walltime=str(aggregate_walltime),
                )
            else:
                action = submitter(
                    run_root=run_root,
                    study_name=study_name,
                    seeds=int(seeds),
                    seed_start=int(seed_start),
                    seed_step=int(seed_step),
                    epochs=int(epochs),
                    max_rounds=int(max_rounds),
                    batch_size=int(batch_size),
                    candidate_pool_min=int(candidate_pool_min),
                    replay_seed_size=int(replay_seed_size),
                    real_strategy=str(real_strategy),
                    strategies=list(strategies),
                    metric=str(metric),
                    target=target,
                    ensemble_size=int(ensemble_size),
                    train_family_for_init=bool(train_family_for_init),
                    use_calibrated_acquisition=bool(use_calibrated_acquisition),
                    generator_objective_mode=str(generator_objective_mode),
                    use_similarity_penalty=bool(use_similarity_penalty),
                    use_length_penalty=bool(use_length_penalty),
                    binary_threshold_strategy=str(binary_threshold_strategy),
                    dry_run=bool(dry_run),
                    force_replay=bool(force_replay),
                    summarize=bool(summarize),
                    allow_config_mismatch=bool(allow_config_mismatch),
                )
        except Exception as exc:
            st.error(str(exc))
            return
        _report_action_result(st, action)


def render_operations_view(st, state: dict[str, object], *, ns: ModuleType) -> None:
    APPROVAL_PENDING_STATUSES = ns.APPROVAL_PENDING_STATUSES
    build_artifact_verification_summary = ns.build_artifact_verification_summary
    _friendly_md_profile = ns._friendly_md_profile
    _frame_empty = ns._frame_empty
    _frame_records = ns._frame_records
    _friendly_bool = ns._friendly_bool
    _multi_metric_chart = ns._multi_metric_chart
    _md_slate_resource_request = ns._md_slate_resource_request
    _notification_rows_for_display = ns._notification_rows_for_display
    _path_name = ns._path_name
    _persisted_choice = ns._persisted_choice
    _quote_path = ns._quote_path
    _render_action_history = ns._render_action_history
    _render_artifact_verification_workspace = ns._render_artifact_verification_workspace
    _render_cluster_health_panel = ns._render_cluster_health_panel
    _render_action_contracts_panel = ns._render_action_contracts_panel
    _render_execution_readiness_panel = ns._render_execution_readiness_panel
    _render_export_pack = ns._render_export_pack
    _render_metric_cards = ns._render_metric_cards
    _render_action_contract_summary = ns._render_action_contract_summary
    _base_render_launch_action = ns._render_launch_action
    _render_operations_md_slate_recovery_center = ns._render_operations_md_slate_recovery_center
    _render_al_loop_simulator_panel = ns._render_al_loop_simulator_panel
    _render_remote_reconciliation_recovery_panel = ns._render_remote_reconciliation_recovery_panel
    _render_remote_console = ns._render_remote_console
    _report_action_result = ns._report_action_result
    _stash_dashboard_flash = ns._stash_dashboard_flash
    _transfer_manifest_rows = ns._transfer_manifest_rows
    _trigger_dashboard_rerun = ns._trigger_dashboard_rerun
    build_run_setup_readiness = ns.build_run_setup_readiness
    coerce_positive_int = ns.coerce_positive_int
    acknowledge_dashboard_notifications = ns.acknowledge_dashboard_notifications
    action_timeline_frame = ns.action_timeline_frame
    action_debug_display_rows = ns.action_debug_display_rows
    build_action_debug_packet_markdown = ns.build_action_debug_packet_markdown
    build_action_debug_rows = ns.build_action_debug_rows
    build_action_debug_summary = ns.build_action_debug_summary
    build_thesis_phase_markdown = ns.build_thesis_phase_markdown
    build_thesis_phase_rows = ns.build_thesis_phase_rows
    build_thesis_phase_summary = ns.build_thesis_phase_summary
    thesis_phase_display_rows = ns.thesis_phase_display_rows
    build_md_slate_monitor_rows = ns.build_md_slate_monitor_rows
    flatten_remote_jobs = ns.flatten_remote_jobs
    hide_dashboard_run = ns.hide_dashboard_run
    list_dashboard_actions = ns.list_dashboard_actions
    pin_dashboard_run = ns.pin_dashboard_run
    draft_supek_submit_study_array_action = ns.draft_supek_submit_study_array_action
    draft_supek_submit_study_action = ns.draft_supek_submit_study_action
    get_cluster_profile = ns.get_cluster_profile
    remote_job_summary_frame = ns.remote_job_summary_frame
    normalize_run_name = ns.normalize_run_name
    parse_strategy_list = ns.parse_strategy_list
    run_setup_defaults = ns.run_setup_defaults
    set_dashboard_run_label = ns.set_dashboard_run_label
    show_dashboard_run = ns.show_dashboard_run
    submit_dashboard_init_run_action = ns.submit_dashboard_init_run_action
    submit_compare_studies_action = ns.submit_compare_studies_action
    submit_run_study_action = ns.submit_run_study_action
    submit_summarize_study_action = ns.submit_summarize_study_action
    unpin_dashboard_run = ns.unpin_dashboard_run
    RUN_SETUP_PRESETS = ns.RUN_SETUP_PRESETS
    STUDY_PRESETS = ns.STUDY_PRESETS
    METRIC_FIELDS = ns.METRIC_FIELDS
    discover_study_manifests = ns.discover_study_manifests
    normalize_study_name = ns.normalize_study_name
    parse_float_or_none = ns.parse_float_or_none
    study_manifest_options = ns.study_manifest_options

    st.header("Operations")
    guided_mode = str(state.get("workflow_mode", "Expert mode")) == "Guided thesis mode"
    approval_actions = [action for action in state.get("actions", []) if action.get("status") in APPROVAL_PENDING_STATUSES]
    remote_actions = [action for action in state.get("actions", []) if action.get("scope") in {"supek", "bura"}]
    snapshots = list(state.get("snapshots", []))
    sync_records = list(state.get("sync_records", []))
    artifact_verification = list(state.get("artifact_verification", []))
    notifications = list(state.get("notifications", []))
    execution_readiness = state.get("execution_readiness", {}) if isinstance(state.get("execution_readiness", {}), dict) else {}
    readiness_counts = execution_readiness.get("counts", {}) if isinstance(execution_readiness.get("counts", {}), dict) else {}
    curation = state.get("curation", {}) if isinstance(state.get("curation", {}), dict) else {}
    action_debug_rows = build_action_debug_rows(list(state.get("actions", [])))
    action_debug_summary = build_action_debug_summary(action_debug_rows)
    thesis_phase_rows = build_thesis_phase_rows(state)
    thesis_phase_summary = build_thesis_phase_summary(thesis_phase_rows)

    _render_metric_cards(
        st,
        [
            ("Configured clusters", sum(1 for row in state["profile_rows"] if str(row.get("configured", "no")) == "yes")),
            ("Unread notifications", sum(1 for row in notifications if str(row.get("state", "")) != "Acknowledged")),
            ("Awaiting approval", len(approval_actions)),
            ("Remote actions", len(remote_actions)),
            ("Action issues", action_debug_summary.get("needs_attention", 0)),
            ("Readiness blocked", readiness_counts.get("blocked", 0)),
            ("Checklist phase", thesis_phase_summary.get("next_phase", "")),
            ("AL simulations", len(list(state.get("al_loop_simulations", [])))),
            ("Pinned runs", len(curation.get("pinned_runs", []))),
        ],
    )
    if guided_mode:
        requested_advanced_section = ""
        session_state = getattr(st, "session_state", {})
        stored_section = str(session_state.get("dashboard_operations_section", "")) if isinstance(session_state, dict) else ""
        radio_values = getattr(st, "_radio_values", {})
        radio_section = str(radio_values.get("dashboard_operations_section", "")) if isinstance(radio_values, dict) else ""
        query_params = getattr(st, "query_params", {})
        query_section = ""
        try:
            query_section = str(query_params.get("operations_section", ""))
        except Exception:
            query_section = ""
        for candidate in [stored_section, radio_section, query_section]:
            if candidate in ADVANCED_OPERATIONS_SECTIONS:
                requested_advanced_section = candidate
                break
        if requested_advanced_section and isinstance(session_state, dict):
            session_state["dashboard_operations_section"] = "Advanced / debug tools"
            session_state["dashboard_operations_advanced_section"] = requested_advanced_section
    section_options = GUIDED_OPERATIONS_SECTIONS if guided_mode else OPERATIONS_SECTIONS
    selected_section = _persisted_choice(
        st,
        st.radio,
        label="Operations section",
        options=section_options,
        key="dashboard_operations_section",
        query_key="operations_section",
        default=section_options[0],
        write_query=False,
    )
    if selected_section == "Advanced / debug tools":
        with st.expander("Advanced / debug tools", expanded=True):
            st.caption("These are still available, just tucked away so normal thesis work does not feel like flying a spacecraft with every maintenance hatch open.")
            selected_section = st.selectbox(
                "Advanced section",
                ADVANCED_OPERATIONS_SECTIONS,
                key="dashboard_operations_advanced_section",
            )

    def _render_launch_action(*args, **kwargs):
        kwargs.setdefault("state", state)
        kwargs.setdefault("view", "Operations")
        kwargs.setdefault("section", selected_section)
        return _base_render_launch_action(*args, **kwargs)

    if selected_section == "Notifications":
        st.subheader("Notifications")
        st.caption("This feed keeps event-style reminders with memory: remote jobs finished, review/ingest handoffs became ready, or cluster health degraded. Acknowledge the items you have handled so the cockpit stays calm.")
        unread_notifications = [row for row in notifications if str(row.get("state", "")) != "Acknowledged"]
        _render_metric_cards(
            st,
            [
                ("Unread / open", len(unread_notifications)),
                ("Acknowledged", sum(1 for row in notifications if str(row.get("state", "")) == "Acknowledged")),
                ("Cluster notices", sum(1 for row in notifications if str(row.get("area", "")) == "Cluster health")),
                ("Review / ingest notices", sum(1 for row in notifications if str(row.get("area", "")) in {"Review ready", "Ingest ready"})),
            ],
        )
        notification_view_options = ["Unread / open", "All current", "Acknowledged only"]
        selected_notification_view = st.selectbox(
            "Notification feed focus",
            notification_view_options,
            index=0,
            key="operations_notification_focus",
        )
        if selected_notification_view == "Unread / open":
            visible_notifications = unread_notifications
        elif selected_notification_view == "Acknowledged only":
            visible_notifications = [row for row in notifications if str(row.get("state", "")) == "Acknowledged"]
        else:
            visible_notifications = notifications
        if visible_notifications:
            if selected_notification_view != "Acknowledged only" and st.button("Acknowledge visible notifications", key="ack_visible_notifications"):
                acknowledge_dashboard_notifications(Path(str(state["run_root"])), [str(row.get("id", "")) for row in visible_notifications])
                _stash_dashboard_flash(st, "success", "Acknowledged the visible notifications.")
                _trigger_dashboard_rerun(st)
            st.dataframe(_notification_rows_for_display({"notifications": visible_notifications}, include_acknowledged=True))
        else:
            st.info("No notifications match the current feed focus.")
        return

    if selected_section == "Thesis checklist":
        st.subheader("Thesis checklist")
        st.caption(
            "This is the live beginner roadmap for the whole thesis system. It is read-only: it audits current artifacts and tells you which dashboard page to open next."
        )
        _render_metric_cards(
            st,
            [
                ("Verdict", thesis_phase_summary.get("verdict", "")),
                ("Progress", f"{thesis_phase_summary.get('progress', 0)}%"),
                ("Done", f"{thesis_phase_summary.get('done', 0)}/{thesis_phase_summary.get('total', 0)}"),
                ("Ready", thesis_phase_summary.get("ready", 0)),
                ("In progress", thesis_phase_summary.get("in_progress", 0)),
                ("Blocked", thesis_phase_summary.get("blocked", 0)),
            ],
        )
        if thesis_phase_summary.get("blocked", 0):
            st.warning("At least one thesis phase is blocked by an action or artifact issue. Start with the safe next move shown below.")
        elif thesis_phase_summary.get("next_phase") == "Complete":
            st.success("The checklist sees packet-ready thesis evidence. Keep the manifests with your thesis exports.")
        else:
            st.info(f"Next click: {thesis_phase_summary.get('next_click', '')}")
        st.markdown("#### Phase table")
        st.dataframe(thesis_phase_display_rows(thesis_phase_rows))
        phase_options = [str(row.get("phase", "")) for row in thesis_phase_rows]
        selected_phase = st.selectbox(
            "Inspect thesis phase",
            phase_options,
            index=0,
            key="operations_thesis_checklist_phase",
        )
        selected_row = next(row for row in thesis_phase_rows if str(row.get("phase", "")) == selected_phase)
        st.markdown(f"#### {selected_row.get('phase', '')}")
        _render_metric_cards(
            st,
            [
                ("Status", selected_row.get("status", "")),
                ("Progress", f"{selected_row.get('progress', 0)}%"),
                ("Open view", selected_row.get("open_view", "")),
                ("Next click", selected_row.get("next_click", "")),
            ],
        )
        st.write(str(selected_row.get("why_it_matters", "")))
        st.info(f"Safe next move: {selected_row.get('safe_next_move', '')}")
        st.caption(f"Evidence: {selected_row.get('evidence', '')}")
        if str(selected_row.get("missing", "")).strip() not in {"", "-"}:
            st.warning(f"Missing: {selected_row.get('missing', '')}")
        if str(selected_row.get("blocker", "")).strip() not in {"", "-"}:
            st.warning(f"Blocker: {selected_row.get('blocker', '')}")
        st.markdown("#### Copy-friendly checklist")
        st.code(build_thesis_phase_markdown(thesis_phase_rows, thesis_phase_summary), language="markdown")
        _render_export_pack(
            st,
            title="Thesis phase checklist export",
            description="Copy-friendly phase rows for lab notes, supervision updates, or thesis planning.",
            rows=thesis_phase_display_rows(thesis_phase_rows),
            key_prefix="operations_thesis_checklist",
        )
        return

    if selected_section == "Action debugger":
        st.subheader("Action debugger")
        st.caption(
            "This is the beginner-friendly failure triage desk. It reads dashboard action state plus stdout/stderr tails, classifies likely causes, and suggests the safest next move before you retry."
        )
        _render_metric_cards(
            st,
            [
                ("Actions inspected", action_debug_summary.get("total", 0)),
                ("Needs fix", action_debug_summary.get("needs_fix", 0)),
                ("Possibly stuck", action_debug_summary.get("possibly_stuck", 0)),
                ("Waiting operator", action_debug_summary.get("waiting_operator", 0)),
                ("Warnings", action_debug_summary.get("needs_review", 0)),
            ],
        )
        focus_options = ["Needs attention", "Failed only", "Stuck / waiting", "All actions"]
        selected_focus = st.selectbox(
            "Debugger focus",
            focus_options,
            index=0,
            key="operations_action_debug_focus",
        )
        if selected_focus == "Failed only":
            visible_debug_rows = [row for row in action_debug_rows if row.get("attention") == "Needs fix"]
        elif selected_focus == "Stuck / waiting":
            visible_debug_rows = [
                row
                for row in action_debug_rows
                if str(row.get("attention", "")) in {"Possibly stuck", "Waiting approval", "Waiting operator"}
            ]
        elif selected_focus == "All actions":
            visible_debug_rows = action_debug_rows
        else:
            visible_debug_rows = [row for row in action_debug_rows if int(row.get("priority", 9)) <= 5]

        if action_debug_summary.get("needs_fix", 0):
            st.warning("At least one action failed and has a suggested safe next move below.")
        elif action_debug_summary.get("possibly_stuck", 0):
            st.info("Some actions may be stale. Inspect logs before cancelling or retrying.")
        elif action_debug_summary.get("needs_attention", 0):
            st.info("There are actions waiting on an operator or warnings worth checking.")
        elif action_debug_rows:
            st.success("No obvious action issues detected.")

        if not visible_debug_rows:
            st.info("No actions match this debugger focus yet.")
            return

        st.markdown("#### Triage table")
        st.dataframe(action_debug_display_rows(visible_debug_rows))
        labels = [
            f"{row.get('attention', '')} | {row.get('title', '')} | {str(row.get('action_id', ''))[-8:]}"
            for row in visible_debug_rows
        ]
        selected_label = st.selectbox(
            "Inspect action diagnosis",
            labels,
            index=0,
            key="operations_action_debug_selected",
        )
        selected_row = visible_debug_rows[labels.index(selected_label)]
        st.markdown("#### Selected diagnosis")
        _render_metric_cards(
            st,
            [
                ("Verdict", selected_row.get("attention", "")),
                ("Issue type", selected_row.get("issue_type", "")),
                ("Status", selected_row.get("status", "")),
                ("Confidence", selected_row.get("confidence", "")),
                ("Target", selected_row.get("target", "")),
            ],
        )
        st.write(str(selected_row.get("reason", "")))
        st.info(f"Safe next move: {selected_row.get('safe_next_move', '')}")
        if str(selected_row.get("evidence", "")).strip():
            st.caption(f"Evidence line: {selected_row.get('evidence', '')}")
        if str(selected_row.get("display_command", "")).strip():
            st.write("Command")
            st.code(str(selected_row.get("display_command", "")), language="bash")
        log_cols = st.columns(2)
        with log_cols[0]:
            st.write("stderr excerpt")
            stderr_excerpt = str(selected_row.get("stderr_excerpt", "")).strip()
            st.code(stderr_excerpt or "No stderr captured.", language="text")
        with log_cols[1]:
            st.write("stdout excerpt")
            stdout_excerpt = str(selected_row.get("stdout_excerpt", "")).strip()
            st.code(stdout_excerpt or "No stdout captured.", language="text")
        st.markdown("#### Debug packet")
        st.caption("Copy this block into lab notes or a support request when you want the exact failure context without digging through JSON files.")
        st.code(build_action_debug_packet_markdown(selected_row), language="markdown")
        _render_export_pack(
            st,
            title="Action debugger table export",
            description="Copy-friendly triage rows for the currently selected debugger focus.",
            rows=action_debug_display_rows(visible_debug_rows),
            key_prefix="operations_action_debugger",
        )
        return

    if selected_section == "Runbook":
        st.subheader("Operator runbook")
        st.caption("Use this as the safe-path guide when you are running real thesis work, debugging a remote job, or deciding what not to touch yet. It is intentionally practical: first move, where to go, next move, and what to avoid.")
        rows = _runbook_rows()
        open_notifications = sum(1 for row in notifications if str(row.get("state", "")) != "Acknowledged")
        stale_notifications = sum(1 for row in notifications if str(row.get("area", "")) == "Stale work" and str(row.get("state", "")) != "Acknowledged")
        artifact_attention = sum(1 for row in artifact_verification if str(row.get("verification_state", "")) == "Attention needed")
        recovery_items = len(list(state.get("md_slate_exceptions", [])))
        active_slates = sum(
            1
            for slate in list(state.get("md_slates", []))
            if str(slate.get("effective_status", "")) not in {"completed", "completed_with_failures", "cancelled"}
        )
        _render_metric_cards(
            st,
            [
                ("Guides", len(rows)),
                ("Open notifications", open_notifications),
                ("Stale reminders", stale_notifications),
                ("Recovery items", recovery_items),
                ("Artifact issues", artifact_attention),
                ("Active slates", active_slates),
            ],
        )
        if stale_notifications:
            st.warning("Some handoffs have been open long enough to become stale. Start with Operations -> Notifications and clear or complete those reminders first.")
        elif artifact_attention or recovery_items or open_notifications:
            st.warning(
                "There is active operator attention in this workspace. Start with Notifications, Recovery center, or Transfers before launching new work."
            )
        else:
            st.success("No active attention blockers are visible in the current workspace. Use the runbook below as the normal safe path.")
        st.write("Quick decision guide")
        st.dataframe(rows)
        guide_options = [str(row["topic"]) for row in rows]
        selected_guide = st.selectbox(
            "Open runbook topic",
            guide_options,
            index=0,
            key="operations_runbook_topic",
        )
        guide = next(row for row in rows if row["topic"] == selected_guide)
        st.markdown(f"#### {guide['topic']}")
        st.info(f"Safe first move: {guide['safe_first_move']}")
        st.write(f"Where to go: {guide['where']}")
        st.write(f"Do next: {guide['do_next']}")
        st.warning(f"Do not do this yet: {guide['do_not']}")
        st.markdown("#### Copy-friendly runbook export")
        _render_export_pack(
            st,
            title="Operator runbook",
            description="Copy this table into lab notes or keep it as the day-to-day operating checklist for the dashboard.",
            rows=rows,
            key_prefix="operations_runbook",
        )
        return

    if selected_section == "New run wizard":
        st.subheader("New run wizard")
        st.caption("Use this when you want to start a thesis experiment without touching the CLI. The wizard creates the run folder, baseline artifacts, optional dashboard pin/label, and can optionally launch replay immediately after init.")
        run_root = Path(str(state["run_root"]))
        all_runs = list(state.get("all_runs", state.get("runs", [])))
        source_options = list(RUN_SETUP_PRESETS)
        if all_runs:
            source_options.append("Clone existing run")
        with _form_container(st, key="new_run_wizard_form"):
            st.caption("Edit the fields, then press Update setup preview. This prevents Streamlit from rerunning the whole page after every single field change.")
            setup_source = st.selectbox(
                "Setup source",
                source_options,
                index=0,
                key="new_run_setup_source",
            )
            clone_run_dir = None
            if setup_source == "Clone existing run":
                clone_options = [
                    str(run.get("run_display_name", run.get("run_name", "")))
                    for run in all_runs
                ]
                selected_clone = st.selectbox(
                    "Clone settings from",
                    clone_options,
                    index=0,
                    key="new_run_clone_source",
                )
                clone_run = next(
                    run
                    for run in all_runs
                    if str(run.get("run_display_name", run.get("run_name", ""))) == selected_clone
                )
                clone_run_dir = Path(str(clone_run.get("run_dir", "")))
            defaults = run_setup_defaults(setup_source, clone_run_dir=clone_run_dir)
            st.info(str(defaults.get("description", "")))
            default_run_name = normalize_run_name(f"{defaults.get('run_name_prefix', 'dashboard_run')}_{len(all_runs) + 1:03d}")
            run_name_raw = st.text_input(
                "Run folder name",
                value=default_run_name,
                key="new_run_name",
                help="Use a short unique folder-style name. Spaces will be converted to underscores before launch.",
            )
            run_name = normalize_run_name(run_name_raw, fallback="")
            if run_name != str(run_name_raw).strip():
                st.caption(f"Normalized run folder: `{run_name}`")
            run_label = st.text_input(
                "Dashboard label",
                value="",
                key="new_run_label",
                help="Optional friendly label shown in the dashboard after the run is created.",
            )
            strategy_defaults = parse_strategy_list(defaults.get("replay_strategies", []))
            strategy_options = list(dict.fromkeys([str(defaults.get("real_strategy", "ensemble_mi")), *strategy_defaults, "random", "family_qbc", "hybrid_mi_diverse"]))
            real_strategy = st.selectbox(
                "Real AL strategy",
                strategy_options,
                index=0,
                key="new_run_real_strategy",
            )
            seed_text = st.text_input("Random seed", value=str(defaults.get("random_seed", 20260317)), key="new_run_seed")
            batch_text = st.text_input("Batch size", value=str(defaults.get("batch_size", 5)), key="new_run_batch_size")
            rounds_text = st.text_input("Max replay / AL rounds", value=str(defaults.get("max_rounds", 10)), key="new_run_max_rounds")
            epochs_text = st.text_input("Training epochs", value=str(defaults.get("epochs", 70)), key="new_run_epochs")
            candidate_pool_text = st.text_input("Candidate pool minimum", value=str(defaults.get("candidate_pool_min", 50)), key="new_run_candidate_pool_min")
            replay_seed_text = st.text_input("Replay seed size", value=str(defaults.get("replay_seed_size", 40)), key="new_run_replay_seed_size")
            replay_strategy_text = st.text_input(
                "Replay strategies",
                value=", ".join(strategy_defaults),
                key="new_run_replay_strategies",
                help="Comma-separated strategy names saved into the run config and used if replay is launched immediately.",
            )
            random_seed = coerce_positive_int(seed_text, int(defaults.get("random_seed", 20260317)), minimum=0)
            batch_size = coerce_positive_int(batch_text, int(defaults.get("batch_size", 5)))
            max_rounds = coerce_positive_int(rounds_text, int(defaults.get("max_rounds", 10)), minimum=0)
            epochs = coerce_positive_int(epochs_text, int(defaults.get("epochs", 70)))
            candidate_pool_min = coerce_positive_int(candidate_pool_text, int(defaults.get("candidate_pool_min", 50)))
            replay_seed_size = coerce_positive_int(replay_seed_text, int(defaults.get("replay_seed_size", 40)))
            replay_strategies = parse_strategy_list(replay_strategy_text)
            train_family_for_init = st.checkbox(
                "Train family committee during init",
                value=bool(defaults.get("train_family_for_init", False)),
                key="new_run_train_family",
            )
            train_baseline_after_init = st.checkbox(
                "Train baseline locally during init",
                value=False,
                key="new_run_train_baseline",
                help="Leave unchecked for a SUPEK-first run. The wizard will create only config/split/ledger files locally, then SUPEK can do the expensive baseline/replay work.",
            )
            use_calibrated_acquisition = st.checkbox(
                "Use calibrated acquisition",
                value=bool(defaults.get("use_calibrated_acquisition", True)),
                key="new_run_calibrated",
            )
            generator_default = str(
                defaults.get(
                    "generator_objective_mode",
                    RunConfig().generator_objective_mode,
                )
            )
            if generator_default not in GENERATOR_OBJECTIVE_MODES:
                generator_default = RunConfig().generator_objective_mode
            generator_objective_mode = st.selectbox(
                "Generator objective mode",
                GENERATOR_OBJECTIVE_MODES,
                index=GENERATOR_OBJECTIVE_MODES.index(generator_default),
                key="new_run_generator_objective_mode",
                help="Use match_acquisition for research-clean real AL; fixed_mean preserves the older practical generator behavior.",
            )
            use_similarity_penalty = st.checkbox(
                "Use similarity penalty during generation",
                value=bool(defaults.get("use_similarity_penalty", False)),
                key="new_run_similarity_penalty",
            )
            use_length_penalty = st.checkbox(
                "Use length penalty during generation",
                value=bool(defaults.get("use_length_penalty", True)),
                key="new_run_length_penalty",
            )
            threshold_default = str(
                defaults.get(
                    "binary_threshold_strategy",
                    RunConfig().binary_threshold_strategy,
                )
            )
            if threshold_default not in BINARY_THRESHOLD_STRATEGIES:
                threshold_default = RunConfig().binary_threshold_strategy
            binary_threshold_strategy = st.selectbox(
                "Binary threshold strategy",
                BINARY_THRESHOLD_STRATEGIES,
                index=BINARY_THRESHOLD_STRATEGIES.index(threshold_default),
                key="new_run_binary_threshold_strategy",
                help="New-run primary F1 is PR-best-F1 thresholded F1; fixed-0.5 metrics remain visible with _fixed_0_5 suffixes.",
            )
            pin_run = st.checkbox(
                "Pin this run into Current Thesis Work",
                value=True,
                key="new_run_pin",
            )
            run_replay_after_init = st.checkbox(
                "Run replay benchmark immediately after init",
                value=False,
                key="new_run_replay_after_init",
            )
            if run_replay_after_init:
                st.info("Replay can take a while. The dashboard queues it as one setup action so you can leave it running and inspect logs from Operations.")
            elif not train_baseline_after_init:
                st.info("SUPEK-first mode: this will create config/split/ledger files only. No local TensorFlow training should start.")
            _form_submit_button(st, "Update setup preview", key="new_run_update_preview")
        preview_rows = [
            {"setting": "Run root", "value": str(run_root)},
            {"setting": "Run folder", "value": run_name},
            {"setting": "Preset/source", "value": setup_source},
            {"setting": "Random seed", "value": random_seed},
            {"setting": "Batch size", "value": batch_size},
            {"setting": "Max rounds", "value": max_rounds},
            {"setting": "Epochs", "value": epochs},
            {"setting": "Candidate pool minimum", "value": candidate_pool_min},
            {"setting": "Replay seed size", "value": replay_seed_size},
            {"setting": "Real strategy", "value": real_strategy},
            {"setting": "Replay strategies", "value": ", ".join(replay_strategies)},
            {"setting": "Generator objective mode", "value": generator_objective_mode},
            {"setting": "Similarity / length penalty", "value": f"{_friendly_bool(use_similarity_penalty)} / {_friendly_bool(use_length_penalty)}"},
            {"setting": "Binary threshold strategy", "value": binary_threshold_strategy},
            {"setting": "Train baseline locally", "value": _friendly_bool(train_baseline_after_init)},
            {"setting": "Pin after create", "value": _friendly_bool(pin_run)},
            {"setting": "Run replay after init", "value": _friendly_bool(run_replay_after_init)},
        ]
        st.markdown("#### Setup preview")
        st.dataframe(preview_rows)
        readiness = build_run_setup_readiness(run_root, run_name=str(run_name_raw))
        setup_command = (
            "python -m active_learning_thesis dashboard-init-run "
            f"--run-root {_quote_path(run_root)} "
            f"--run-name {run_name} "
            f"--random-seed {random_seed} "
            f"--batch-size {batch_size} "
            f"--max-rounds {max_rounds} "
            f"--epochs {epochs} "
            f"--candidate-pool-min {candidate_pool_min} "
            f"--replay-seed-size {replay_seed_size} "
            f"--real-strategy {real_strategy} "
            f'--replay-strategies "{",".join(replay_strategies)}" '
            f"--generator-objective-mode {generator_objective_mode} "
            f"--binary-threshold-strategy {binary_threshold_strategy}"
        )
        if train_family_for_init:
            setup_command += " --train-family-for-init"
        if not use_calibrated_acquisition:
            setup_command += " --raw-acquisition"
        if use_similarity_penalty:
            setup_command += " --use-similarity-penalty"
        if not use_length_penalty:
            setup_command += " --no-length-penalty"
        if pin_run:
            setup_command += " --pin-run"
        if str(run_label).strip():
            setup_command += f' --run-label "{str(run_label).replace(chr(34), "").strip()}"'
        if not train_baseline_after_init:
            setup_command += " --skip-baseline-init"
        if run_replay_after_init:
            setup_command += " --run-replay-after-init"
        _render_launch_action(
            st,
            label="Create new thesis run",
            command=setup_command,
            key_prefix="new_run_wizard",
            button_text="Create run",
            what="Create the run directory, save config/split/ledger artifacts, optionally train the baseline ensemble, optionally pin and label the run, and optionally run replay immediately.",
            when="Use this when starting a new thesis experiment or cloning settings from a previous run.",
            produces="A new run folder that appears in Model Workflow and Current Thesis Work after refresh. If baseline training is skipped, the run remains config-only until a local or remote workflow trains it.",
            next_step="Open Model Workflow for the new run and continue with replay, proposal, MD validation, or reporting depending on the selected options.",
            contract_id="init-run",
            readiness=readiness,
            on_submit=lambda run_name=run_name, random_seed=random_seed, batch_size=batch_size, max_rounds=max_rounds, epochs=epochs, candidate_pool_min=candidate_pool_min, replay_seed_size=replay_seed_size, real_strategy=real_strategy, replay_strategies=replay_strategies, train_family_for_init=train_family_for_init, train_baseline_after_init=train_baseline_after_init, use_calibrated_acquisition=use_calibrated_acquisition, generator_objective_mode=generator_objective_mode, use_similarity_penalty=use_similarity_penalty, use_length_penalty=use_length_penalty, binary_threshold_strategy=binary_threshold_strategy, pin_run=pin_run, run_label=run_label, run_replay_after_init=run_replay_after_init: submit_dashboard_init_run_action(
                run_root=run_root,
                run_name=run_name,
                random_seed=int(random_seed),
                batch_size=int(batch_size),
                max_rounds=int(max_rounds),
                epochs=int(epochs),
                candidate_pool_min=int(candidate_pool_min),
                replay_seed_size=int(replay_seed_size),
                real_strategy=str(real_strategy),
                replay_strategies=list(replay_strategies),
                train_family_for_init=bool(train_family_for_init),
                train_baseline_after_init=bool(train_baseline_after_init),
                use_calibrated_acquisition=bool(use_calibrated_acquisition),
                generator_objective_mode=str(generator_objective_mode),
                use_similarity_penalty=bool(use_similarity_penalty),
                use_length_penalty=bool(use_length_penalty),
                binary_threshold_strategy=str(binary_threshold_strategy),
                pin_run=bool(pin_run),
                run_label=str(run_label),
                run_replay_after_init=bool(run_replay_after_init),
            ),
        )
        return

    if selected_section == "Study designer":
        st.subheader("Study designer")
        st.caption("Use this when your thesis needs evidence across seeds, strategies, or ablation settings. These actions run in the dashboard queue and write under `_studies/` or `_study_evidence/`.")
        run_root = Path(str(state["run_root"]))
        study_rows = discover_study_manifests(run_root)
        _render_metric_cards(
            st,
            [
                ("Study manifests", len(study_rows)),
                ("Completed studies", sum(1 for row in study_rows if str(row.get("status", "")) == "completed")),
                ("Failed / partial", sum(1 for row in study_rows if str(row.get("status", "")) in {"failed", "partial"})),
                ("Available comparisons", len(study_manifest_options(run_root))),
            ],
        )
        mode = st.selectbox(
            "Study task",
            ["Run study / ablation", "Summarize replay evidence", "Compare studies"],
            index=0,
            key="study_designer_mode",
        )

        if mode == "Run study / ablation":
            st.info(
                "This editor is a form: change as many study fields as you want, then click the final queue/draft button once. "
                "The dashboard should not rerun on every individual field edit."
            )
            with _form_container(st, key="study_designer_run_ablation_form"):
                _render_study_run_ablation_form(
                    st,
                    state=state,
                    run_root=run_root,
                    study_rows=study_rows,
                    STUDY_PRESETS=STUDY_PRESETS,
                    METRIC_FIELDS=METRIC_FIELDS,
                    coerce_positive_int=coerce_positive_int,
                    normalize_study_name=normalize_study_name,
                    parse_float_or_none=parse_float_or_none,
                    parse_strategy_list=parse_strategy_list,
                    _friendly_bool=_friendly_bool,
                    _quote_path=_quote_path,
                    get_cluster_profile=get_cluster_profile,
                    build_button_readiness_contract=_render_action_contract_summary,
                    submit_run_study_action=submit_run_study_action,
                    draft_supek_submit_study_action=draft_supek_submit_study_action,
                    draft_supek_submit_study_array_action=draft_supek_submit_study_array_action,
                    list_dashboard_actions=list_dashboard_actions,
                    _render_action_history=_render_action_history,
                    _report_action_result=_report_action_result,
                    APPROVAL_PENDING_STATUSES=APPROVAL_PENDING_STATUSES,
                )
            return

        if mode == "__legacy_run_study_ablation":
            preset_name = st.selectbox(
                "Study preset",
                list(STUDY_PRESETS),
                index=0,
                key="study_designer_preset",
            )
            preset = dict(STUDY_PRESETS[preset_name])
            st.info(str(preset.get("description", "")))
            default_study_name = normalize_study_name(f"{preset.get('study_name_prefix', 'study')}_{len(study_rows) + 1:03d}")
            raw_study_name = st.text_input("Study name", value=default_study_name, key="study_designer_name")
            study_name = normalize_study_name(raw_study_name, fallback="")
            if study_name != str(raw_study_name).strip():
                st.caption(f"Normalized study name: `{study_name}`")
            metric = st.selectbox(
                "Primary metric",
                METRIC_FIELDS,
                index=METRIC_FIELDS.index(str(preset.get("metric", "f1"))) if str(preset.get("metric", "f1")) in METRIC_FIELDS else 0,
                key="study_designer_metric",
            )
            target_text = st.text_input("Optional target metric", value=str(preset.get("target", "")), key="study_designer_target")
            seeds = coerce_positive_int(st.text_input("Seed count", value=str(preset.get("seeds", 5)), key="study_designer_seeds"), int(preset.get("seeds", 5)))
            seed_start = coerce_positive_int(st.text_input("Seed start", value=str(preset.get("seed_start", 20260317)), key="study_designer_seed_start"), int(preset.get("seed_start", 20260317)), minimum=0)
            seed_step = coerce_positive_int(st.text_input("Seed step", value=str(preset.get("seed_step", 1009)), key="study_designer_seed_step"), int(preset.get("seed_step", 1009)))
            epochs = coerce_positive_int(st.text_input("Epochs", value=str(preset.get("epochs", 70)), key="study_designer_epochs"), int(preset.get("epochs", 70)))
            max_rounds = coerce_positive_int(st.text_input("Max rounds", value=str(preset.get("max_rounds", 10)), key="study_designer_max_rounds"), int(preset.get("max_rounds", 10)), minimum=0)
            batch_size = coerce_positive_int(st.text_input("Batch size", value=str(preset.get("batch_size", 5)), key="study_designer_batch_size"), int(preset.get("batch_size", 5)))
            candidate_pool_min = coerce_positive_int(st.text_input("Candidate pool minimum", value=str(preset.get("candidate_pool_min", 50)), key="study_designer_candidate_pool"), int(preset.get("candidate_pool_min", 50)))
            replay_seed_size = coerce_positive_int(st.text_input("Replay seed size", value=str(preset.get("replay_seed_size", 40)), key="study_designer_replay_seed"), int(preset.get("replay_seed_size", 40)))
            ensemble_size = coerce_positive_int(st.text_input("Ensemble size", value=str(preset.get("ensemble_size", 5)), key="study_designer_ensemble_size"), int(preset.get("ensemble_size", 5)))
            strategy_text = st.text_input(
                "Replay strategies",
                value=", ".join(str(item) for item in list(preset.get("strategies", []))),
                key="study_designer_strategies",
            )
            strategies = parse_strategy_list(strategy_text)
            real_strategy_options = list(dict.fromkeys([str(preset.get("real_strategy", "ensemble_mi")), *strategies, "ensemble_mi", "family_qbc"]))
            real_strategy = st.selectbox(
                "Real AL strategy",
                real_strategy_options,
                index=0,
                key="study_designer_real_strategy",
            )
            train_family_for_init = st.checkbox("Train family committee during init", value=bool(preset.get("train_family_for_init", False)), key="study_designer_train_family")
            use_calibrated_acquisition = st.checkbox("Use calibrated acquisition", value=bool(preset.get("use_calibrated_acquisition", True)), key="study_designer_calibrated")
            run_on_supek = st.checkbox("Submit real study to SUPEK instead of local dashboard queue", value=False, key="study_designer_run_on_supek")
            supek_walltime = ""
            supek_profile = get_cluster_profile(state.get("profiles", {}), "supek") if run_on_supek else None
            if run_on_supek:
                default_walltime = str((supek_profile or {}).get("default_walltime", "12:00:00") or "12:00:00")
                supek_walltime = st.text_input("SUPEK walltime", value=default_walltime if default_walltime != "01:00:00" else "12:00:00", key="study_designer_supek_walltime")
                split_supek_by_seed = st.checkbox(
                    "Split SUPEK study into one job per seed plus aggregate job",
                    value=int(seeds) > 1,
                    key="study_designer_split_supek_by_seed",
                )
                aggregate_walltime = st.text_input(
                    "SUPEK aggregate walltime",
                    value="02:00:00",
                    key="study_designer_supek_aggregate_walltime",
                    help="Small dependency job after all seed jobs finish. It rebuilds the full study manifest and summary.",
                ) if split_supek_by_seed else ""
                if supek_profile is None:
                    st.warning("SUPEK profile is not configured yet, so remote study submission is blocked.")
            else:
                split_supek_by_seed = False
                aggregate_walltime = ""
            dry_run = st.checkbox("Dry run only: write study plan without training", value=True, key="study_designer_dry_run")
            force_replay = st.checkbox("Force replay even if summaries exist", value=False, key="study_designer_force_replay")
            summarize = st.checkbox("Summarize automatically after study completes", value=True, key="study_designer_summarize")
            allow_config_mismatch = st.checkbox("Allow config mismatch when resuming", value=False, key="study_designer_allow_mismatch")
            target = parse_float_or_none(target_text)
            target_text_invalid = bool(str(target_text).strip()) and target is None
            if target_text_invalid:
                st.warning("Target must be numeric before the study can be queued.")
            preview_rows = [
                {"setting": "Study name", "value": study_name},
                {"setting": "Run root", "value": str(run_root)},
                {"setting": "Seeds", "value": seeds},
                {"setting": "Seed start / step", "value": f"{seed_start} / {seed_step}"},
                {"setting": "Metric / target", "value": f"{metric} / {target if target is not None else '-'}"},
                {"setting": "Strategies", "value": ", ".join(strategies)},
                {"setting": "Expected run folders", "value": seeds},
                {"setting": "Dry run", "value": _friendly_bool(dry_run)},
                {"setting": "Execution target", "value": f"SUPEK ({supek_walltime or '-'})" if run_on_supek else "Local dashboard queue"},
                {"setting": "SUPEK split mode", "value": "per-seed + aggregate" if run_on_supek and split_supek_by_seed else "-"},
                {"setting": "Auto-summarize", "value": _friendly_bool(summarize)},
            ]
            st.markdown("#### Study plan preview")
            st.dataframe(preview_rows)
            study_command = (
                "python -m active_learning_thesis run-study "
                f"--study-name {study_name} "
                f"--run-root {_quote_path(run_root)} "
                f"--seeds {seeds} --seed-start {seed_start} --seed-step {seed_step} "
                f"--epochs {epochs} --max-rounds {max_rounds} --batch-size {batch_size} "
                f"--candidate-pool-min {candidate_pool_min} --replay-seed-size {replay_seed_size} "
                f"--real-strategy {real_strategy} --ensemble-size {ensemble_size} --metric {metric}"
            )
            if strategies:
                study_command += " --strategies " + " ".join(strategies)
            if target is not None:
                study_command += f" --target {target}"
            if train_family_for_init:
                study_command += " --train-family-for-init"
            if not use_calibrated_acquisition:
                study_command += " --raw-acquisition"
            if dry_run:
                study_command += " --dry-run"
            if force_replay:
                study_command += " --force-replay"
            if not summarize:
                study_command += " --no-summarize"
            if allow_config_mismatch:
                study_command += " --allow-config-mismatch"
            study_blockers = []
            study_cautions = []
            if not study_name:
                study_blockers.append("Study name is empty.")
            if not strategies:
                study_blockers.append("At least one replay strategy is required.")
            if target_text_invalid:
                study_blockers.append("Target metric must be a finite number or blank.")
            if run_on_supek and supek_profile is None:
                study_blockers.append("SUPEK profile is not configured.")
            if run_on_supek and dry_run:
                study_cautions.append("Dry-run mode is enabled; the SUPEK job will only write a remote study plan and will not train.")
            if run_on_supek and split_supek_by_seed:
                study_cautions.append("Split mode submits one independent seed job per seed plus one dependency aggregate job. This is faster and safer against timeout, but uses more queue slots.")
            if study_name and (run_root / "_studies" / study_name).exists():
                study_cautions.append("Study directory already exists; this action will resume or update the existing manifest.")
            if not run_root.exists():
                study_cautions.append(f"Run root will be created: {run_root}")
            study_readiness = {
                "verdict": "Blocked" if study_blockers else ("Ready with caution" if study_cautions else "Ready"),
                "summary": "Review the study plan before queueing it.",
                "blockers": study_blockers,
                "cautions": study_cautions,
                "fix_now": "Use a non-empty study name, at least one replay strategy, and a numeric target if provided.",
                "disable_button": bool(study_blockers),
            }
            if run_on_supek:
                submitter = draft_supek_submit_study_array_action if split_supek_by_seed else draft_supek_submit_study_action
                button_text = "Draft split SUPEK study jobs" if split_supek_by_seed else "Draft SUPEK study job"
                contract_id = "supek-submit-study-array" if split_supek_by_seed else "supek-submit-study"
                _render_launch_action(
                    st,
                    label="Submit split study on SUPEK" if split_supek_by_seed else "Submit study on SUPEK",
                    command=study_command.replace(f"--run-root {_quote_path(run_root)}", f"--run-root {str((supek_profile or {}).get('scratch_run_root', '<supek scratch>'))}"),
                    key_prefix="study_designer_submit_supek_study",
                    button_text=button_text,
                    what="Create approval-gated SUPEK PBS wrappers that run the selected multi-seed study under the remote scratch run root.",
                    when="Use this after the dry-run plan looks correct and you want real training/replay compute on SUPEK instead of Windows.",
                    produces="A SUPEK dashboard action draft with PBS wrapper paths, stdout/stderr paths, walltime, and tracked job metadata after approval.",
                    next_step="Approve the draft, then monitor it from Operations -> Remote jobs and later pull the study artifacts back.",
                    contract_id=contract_id,
                    readiness=study_readiness,
                    on_submit=lambda submitter=submitter, study_name=study_name, seeds=seeds, seed_start=seed_start, seed_step=seed_step, epochs=epochs, max_rounds=max_rounds, batch_size=batch_size, candidate_pool_min=candidate_pool_min, replay_seed_size=replay_seed_size, real_strategy=real_strategy, strategies=strategies, metric=metric, target=target, ensemble_size=ensemble_size, train_family_for_init=train_family_for_init, use_calibrated_acquisition=use_calibrated_acquisition, dry_run=dry_run, force_replay=force_replay, summarize=summarize, allow_config_mismatch=allow_config_mismatch, supek_walltime=supek_walltime, aggregate_walltime=aggregate_walltime: submitter(
                        run_root=run_root,
                        study_name=study_name,
                        profile=supek_profile,
                        seeds=int(seeds),
                        seed_start=int(seed_start),
                        seed_step=int(seed_step),
                        epochs=int(epochs),
                        max_rounds=int(max_rounds),
                        batch_size=int(batch_size),
                        candidate_pool_min=int(candidate_pool_min),
                        replay_seed_size=int(replay_seed_size),
                        real_strategy=str(real_strategy),
                        strategies=list(strategies),
                        metric=str(metric),
                        target=target,
                        ensemble_size=int(ensemble_size),
                        train_family_for_init=bool(train_family_for_init),
                        use_calibrated_acquisition=bool(use_calibrated_acquisition),
                        dry_run=bool(dry_run),
                        force_replay=bool(force_replay),
                        summarize=bool(summarize),
                        allow_config_mismatch=bool(allow_config_mismatch),
                        walltime=str(supek_walltime),
                        aggregate_walltime=str(aggregate_walltime),
                    ),
                )
                fresh_actions = list_dashboard_actions(run_root)
                pending_study_drafts = [
                    action
                    for action in fresh_actions
                    if str(action.get("kind", "")) in {"supek-submit-study", "supek-submit-study-array"}
                    and str(action.get("status", "")) in APPROVAL_PENDING_STATUSES
                    and str(action.get("metadata", {}).get("study_name", "")) == study_name
                ]
                if pending_study_drafts:
                    st.markdown("#### Pending SUPEK study approval")
                    st.info("This study already has a SUPEK draft waiting. You can approve it here without opening the global Approval queue.")
                    _render_action_history(
                        st,
                        actions=pending_study_drafts,
                        run_root=str(run_root),
                        key_prefix="study_designer_pending_supek_study",
                    )
            else:
                _render_launch_action(
                    st,
                    label="Run study / ablation",
                    command=study_command,
                    key_prefix="study_designer_run_study",
                    button_text="Queue study",
                    what="Create or resume the requested per-seed study runs, run replay strategies, and optionally summarize the resulting evidence.",
                    when="Use this when you need thesis evidence that is stronger than a single run.",
                    produces="A study manifest under `_studies/<study>/` plus replay evidence and optional strategy summary tables.",
                    next_step="Open Results -> Study comparison hub to inspect the summary or compare this study against another.",
                    contract_id="run-study",
                    readiness=study_readiness,
                    on_submit=lambda study_name=study_name, seeds=seeds, seed_start=seed_start, seed_step=seed_step, epochs=epochs, max_rounds=max_rounds, batch_size=batch_size, candidate_pool_min=candidate_pool_min, replay_seed_size=replay_seed_size, real_strategy=real_strategy, strategies=strategies, metric=metric, target=target, ensemble_size=ensemble_size, train_family_for_init=train_family_for_init, use_calibrated_acquisition=use_calibrated_acquisition, dry_run=dry_run, force_replay=force_replay, summarize=summarize, allow_config_mismatch=allow_config_mismatch: submit_run_study_action(
                        run_root=run_root,
                        study_name=study_name,
                        seeds=int(seeds),
                        seed_start=int(seed_start),
                        seed_step=int(seed_step),
                        epochs=int(epochs),
                        max_rounds=int(max_rounds),
                        batch_size=int(batch_size),
                        candidate_pool_min=int(candidate_pool_min),
                        replay_seed_size=int(replay_seed_size),
                        real_strategy=str(real_strategy),
                        strategies=list(strategies),
                        metric=str(metric),
                        target=target,
                        ensemble_size=int(ensemble_size),
                        train_family_for_init=bool(train_family_for_init),
                        use_calibrated_acquisition=bool(use_calibrated_acquisition),
                        dry_run=bool(dry_run),
                        force_replay=bool(force_replay),
                        summarize=bool(summarize),
                        allow_config_mismatch=bool(allow_config_mismatch),
                    ),
                )
            return

        if mode == "Summarize replay evidence":
            metric = st.selectbox("Summary metric", METRIC_FIELDS, index=METRIC_FIELDS.index("f1") if "f1" in METRIC_FIELDS else 0, key="study_summary_metric")
            target_text = st.text_input("Optional target metric", value="", key="study_summary_target")
            output_text = st.text_input("Output directory (blank = default)", value="", key="study_summary_output")
            target = parse_float_or_none(target_text)
            target_text_invalid = bool(str(target_text).strip()) and target is None
            if target_text_invalid:
                st.warning("Target must be numeric before the summary can be queued.")
            output_dir = Path(output_text) if str(output_text).strip() else None
            summary_command = f"python -m active_learning_thesis summarize-study --run-root {_quote_path(run_root)} --metric {metric}"
            if target is not None:
                summary_command += f" --target {target}"
            if output_dir is not None:
                summary_command += f" --output-dir {_quote_path(output_dir)}"
            summary_readiness = {
                "verdict": "Blocked" if target_text_invalid else "Ready",
                "summary": "Ready to aggregate replay evidence for the selected metric.",
                "blockers": ["Target metric must be a finite number or blank."] if target_text_invalid else [],
                "cautions": [],
                "fix_now": "Correct the target metric or leave it blank.",
                "disable_button": bool(target_text_invalid),
            }
            _render_launch_action(
                st,
                label="Summarize study evidence",
                command=summary_command,
                key_prefix="study_designer_summarize",
                button_text="Queue summary",
                what="Aggregate replay-capable runs into run/strategy, paired-vs-random, and strategy-level summary tables.",
                when="Use this after replay outputs exist, or when you want a fresh metric-specific summary.",
                produces="Study evidence CSVs and a summary JSON under `_study_evidence/` or your chosen output directory.",
                next_step="Open Results -> Study comparison hub to inspect the best strategy and copy-ready tables.",
                contract_id="summarize-study",
                readiness=summary_readiness,
                on_submit=lambda metric=metric, target=target, output_dir=output_dir: submit_summarize_study_action(
                    run_root=run_root,
                    metric=str(metric),
                    target=target,
                    output_dir=output_dir,
                ),
            )
            return

        manifest_options = study_manifest_options(run_root)
        if len(manifest_options) < 2:
            st.info("At least two study manifests are needed before the comparison action is useful. Run or dry-run two studies first.")
            if study_rows:
                st.dataframe(study_rows)
            return
        baseline = st.selectbox("Baseline study manifest", manifest_options, index=0, key="study_compare_baseline")
        candidate = st.selectbox("Candidate study manifest", manifest_options, index=1 if len(manifest_options) > 1 else 0, key="study_compare_candidate")
        metric = st.selectbox("Comparison metric", METRIC_FIELDS, index=METRIC_FIELDS.index("f1") if "f1" in METRIC_FIELDS else 0, key="study_compare_metric")
        target_text = st.text_input("Optional target metric", value="", key="study_compare_target")
        output_text = st.text_input("Output directory (blank = default)", value="", key="study_compare_output")
        target = parse_float_or_none(target_text)
        target_text_invalid = bool(str(target_text).strip()) and target is None
        output_dir = Path(output_text) if str(output_text).strip() else None
        compare_command = (
            "python -m active_learning_thesis compare-studies "
            f"--run-root {_quote_path(run_root)} "
            f"--baseline-study {_quote_path(baseline)} "
            f"--candidate-study {_quote_path(candidate)} "
            f"--metric {metric}"
        )
        if target is not None:
            compare_command += f" --target {target}"
        if output_dir is not None:
            compare_command += f" --output-dir {_quote_path(output_dir)}"
        same_manifest = Path(baseline).resolve() == Path(candidate).resolve()
        compare_blockers = []
        if same_manifest:
            compare_blockers.append("Choose two different study manifests.")
        if target_text_invalid:
            compare_blockers.append("Target metric must be a finite number or blank.")
        if target_text_invalid:
            st.warning("Target must be numeric before the comparison can be queued.")
        compare_readiness = {
            "verdict": "Blocked" if compare_blockers else "Ready",
            "summary": "Ready to compare the selected study manifests.",
            "blockers": compare_blockers,
            "cautions": [],
            "fix_now": "Pick two different manifests and correct the target metric if provided.",
            "disable_button": bool(compare_blockers),
        }
        _render_launch_action(
            st,
            label="Compare studies",
            command=compare_command,
            key_prefix="study_designer_compare",
            button_text="Queue comparison",
            what="Compare two matched study manifests by seed/strategy and write paired deltas plus a thesis narrative.",
            when="Use this for ablations such as calibrated vs raw acquisition, or two study presets with matching seeds and strategies.",
            produces="A comparison bundle under `_studies/_comparisons/` with paired rows, strategy deltas, summary JSON, and narrative markdown.",
            next_step="Open Results -> Study comparison hub to inspect the matched pairs and strongest deltas.",
            contract_id="compare-studies",
            readiness=compare_readiness,
            on_submit=lambda baseline=baseline, candidate=candidate, metric=metric, target=target, output_dir=output_dir: submit_compare_studies_action(
                run_root=run_root,
                baseline_study=str(baseline),
                candidate_study=str(candidate),
                metric=str(metric),
                target=target,
                output_dir=output_dir,
            ),
        )
        return

    if selected_section == "Action contracts":
        _render_action_contracts_panel(st, state)
        return

    if selected_section == "Execution readiness":
        _render_execution_readiness_panel(st, state)
        return

    if selected_section == "AL loop simulator":
        _render_al_loop_simulator_panel(st, state)
        return

    if selected_section == "Cluster health":
        st.subheader("Cluster profiles")
        st.write(f"Profile config path: `{state['profiles']['path']}`")
        st.dataframe(state["profile_rows"])
        st.info(
            "Cluster profiles unlock connectivity only. Use Model Workflow for SUPEK actions and MD Validation for BURA actions. Buttons appear only when the selected run or peptide is at the right step."
        )
        _render_cluster_health_panel(st, state)
        if not state["profiles"]["exists"]:
            st.warning("No cluster profile file found yet. Use this template:")
            st.json(state["profiles"]["template"])
        return

    if selected_section == "Approval queue":
        st.subheader("Approval queue")
        fresh_actions = list_dashboard_actions(Path(str(state["run_root"])))
        fresh_approval_actions = [
            action
            for action in fresh_actions
            if str(action.get("status", "")) in APPROVAL_PENDING_STATUSES
        ]
        visible_approval_actions = fresh_approval_actions or approval_actions
        st.caption(f"Pending approval actions found: {len(visible_approval_actions)}")
        if visible_approval_actions:
            _render_action_history(
                st,
                actions=visible_approval_actions,
                run_root=str(state["run_root"]),
                key_prefix="operations_approval",
            )
        else:
            st.info("No draft actions are waiting on approval.")
        return

    if selected_section == "Remote jobs":
        st.subheader("Remote job inventory")
        md_slates = list(state.get("md_slates", []))
        active_md_slates = [
            slate
            for slate in md_slates
            if str(slate.get("effective_status", "")) not in {"completed", "completed_with_failures", "cancelled"}
        ]
        bura_utilization = state.get("bura_utilization", {}) if isinstance(state.get("bura_utilization", {}), dict) else {}
        snapshot_summary = bura_utilization.get("snapshot_summary", {}) if isinstance(bura_utilization, dict) else {}
        tracked_counts = bura_utilization.get("tracked_external_counts", {}) if isinstance(bura_utilization, dict) else {}
        remote_reconciliation = list(state.get("remote_reconciliation", []))
        remote_reconciliation_summary = (
            state.get("remote_reconciliation_summary", {})
            if isinstance(state.get("remote_reconciliation_summary", {}), dict)
            else {}
        )
        remote_watchdog = list(state.get("remote_watchdog", []))
        remote_watchdog_summary = (
            state.get("remote_watchdog_summary", {})
            if isinstance(state.get("remote_watchdog_summary", {}), dict)
            else {}
        )
        if active_md_slates or tracked_counts:
            st.markdown("#### MD slate orchestration")
            _render_metric_cards(
                st,
                [
                    ("Active slates", len(active_md_slates)),
                    ("Tracked line_smoke", tracked_counts.get("line_smoke", 0)),
                    ("Tracked production_smoke", tracked_counts.get("production_smoke", 0)),
                    ("Tracked full", tracked_counts.get("full", 0)),
                    ("BURA running", snapshot_summary.get("running", 0)),
                    ("BURA pending", snapshot_summary.get("pending", 0)),
                    ("BURA held", snapshot_summary.get("held", 0)),
                ],
            )
        if remote_watchdog:
            st.markdown("#### Remote heartbeat autopilot")
            _render_metric_cards(
                st,
                [
                    ("Tracked items", remote_watchdog_summary.get("total", 0)),
                    ("Needs recovery", remote_watchdog_summary.get("needs_recovery", 0)),
                    ("Needs check", remote_watchdog_summary.get("needs_check", 0)),
                    ("Watching", remote_watchdog_summary.get("watch", 0)),
                    ("Ready", remote_watchdog_summary.get("ready", 0)),
                    ("Read-only follow-ups", remote_watchdog_summary.get("read_only_followups", 0)),
                ],
            )
            if remote_watchdog_summary.get("needs_recovery", 0):
                st.warning("The watchdog found remote drift that should be reconciled before launching more work.")
            elif remote_watchdog_summary.get("needs_check", 0):
                st.info("The watchdog wants fresh read-only evidence, such as queue polling, logs, or health/readiness checks.")
            else:
                st.success("Tracked remote work has a coherent heartbeat in the latest local evidence.")
            st.caption("This panel is read-only. It turns queue snapshots, health checks, slate state, sync records, reconciliation, and artifact checks into one verdict per tracked remote item.")
            st.dataframe(_remote_watchdog_display_rows(remote_watchdog))
        _render_remote_reconciliation_recovery_panel(
            st,
            state,
            remote_reconciliation,
            remote_reconciliation_summary,
        )
        if active_md_slates:
            st.caption("These are the dashboard-managed MD slates currently in flight. The table below shows which run they belong to and how many peptides are active, blocked, or already review-ready.")
            st.dataframe(
                [
                    {
                        "run": _path_name(str(row.get("run_dir", ""))),
                        "slate_id": row.get("slate_id", ""),
                        "mode": row.get("execution_mode", "live"),
                        "status": row.get("effective_status", ""),
                        "peptides": row.get("peptide_count", 0),
                        "active": row.get("active_count", 0),
                        "blocked": row.get("blocked_count", 0),
                        "exceptions": row.get("exception_count", 0),
                        "review_ready": row.get("review_ready_count", 0),
                    }
                    for row in active_md_slates
                ]
            )
            slate_options = [
                f"{_path_name(str(item.get('run_dir', '')))} [{str(item.get('slate_id', ''))}]"
                for item in active_md_slates
            ]
            selected_slate_label = st.selectbox(
                "Inspect active MD slate",
                slate_options,
                index=0,
                key="operations_md_slate_select",
            )
            selected_slate = active_md_slates[slate_options.index(selected_slate_label)]
            st.dataframe(
                [
                    {
                        **row,
                        "stage_label": _friendly_md_profile(str(row.get("stage", ""))) if str(row.get("stage", "")) not in {"", "-"} else "-",
                        "resource_request": _md_slate_resource_request(str(row.get("stage", ""))),
                    }
                    for row in build_md_slate_monitor_rows(selected_slate)
                ]
            )
        tracked_rows = bura_utilization.get("tracked_external_rows", []) if isinstance(bura_utilization, dict) else []
        if tracked_rows:
            st.caption("Tracked active BURA jobs outside the selected slate still count toward the local 2 / 1 / 1 safety caps. Unknown external jobs are only shown in the queue summary below.")
            st.dataframe(tracked_rows)
        if remote_actions:
            st.dataframe(action_timeline_frame(remote_actions))
        else:
            st.info("No remote actions have been recorded yet.")
        if remote_actions:
            st.markdown("#### Live cluster consoles")
            console_cols = st.columns(2)
            with console_cols[0]:
                _render_remote_console(
                    st,
                    title="SUPEK live console",
                    actions=remote_actions,
                    kinds={
                        "supek-verify-env",
                        "supek-sync-repo",
                        "supek-sync-run",
                        "supek-submit-preflight",
                        "supek-poll-qstat",
                        "supek-fetch-logs",
                        "supek-submit-workflow",
                        "supek-cancel-job",
                        "supek-pull-artifacts",
                    },
                    key_prefix="operations_supek_console",
                )
            with console_cols[1]:
                _render_remote_console(
                    st,
                    title="BURA live console",
                    actions=remote_actions,
                    kinds={"bura-submit-readiness", "bura-poll-squeue", "bura-inspect-logs", "bura-submit-chain", "bura-cancel-job", "bura-preflight"},
                    key_prefix="operations_bura_console",
                )
        if snapshots:
            summary_frame = remote_job_summary_frame(snapshots)
            st.dataframe(summary_frame)
            summary_rows = _frame_records(summary_frame)
            if summary_rows:
                chart = _multi_metric_chart(
                    summary_rows,
                    index_key="cluster",
                    value_keys=["pending", "running", "failed", "held"],
                    label_key="cluster",
                )
                if chart:
                    st.bar_chart(chart)
            flat_jobs = flatten_remote_jobs(snapshots)
            if not _frame_empty(flat_jobs):
                st.markdown("#### Recent remote jobs")
                st.dataframe(flat_jobs)
        else:
            st.info("No cluster snapshots have been captured yet.")
        return

    if selected_section == "Recovery center":
        _render_operations_md_slate_recovery_center(st, state)
        return

    if selected_section == "Transfers":
        st.subheader("Transfer and sync status")
        if sync_records:
            manifest_rows = _transfer_manifest_rows(sync_records)
            artifact_attention = [
                row for row in artifact_verification
                if str(row.get("verification_state", "")) == "Attention needed"
            ]
            artifact_summary = build_artifact_verification_summary(artifact_verification)
            _render_metric_cards(
                st,
                [
                    ("Tracked transfers", len(manifest_rows)),
                    ("Uploaded / remote active", sum(1 for row in manifest_rows if str(row.get("direction", "")) == "Local -> remote")),
                    ("Downloaded / staged", sum(1 for row in manifest_rows if str(row.get("transfer_state", "")) == "Downloaded into safe staging")),
                    ("Finalized locally", sum(1 for row in manifest_rows if str(row.get("transfer_state", "")) == "Finalized locally")),
                    ("Artifact issues", artifact_summary.get("attention", 0)),
                ],
            )
            st.caption("Use this manifest when you want to trust the remote/local handoff. It shows what moved, where it is now, and what the safest next action is.")
            st.dataframe(manifest_rows)
            st.markdown("#### Transfer manifest export")
            _render_export_pack(
                st,
                title="Transfer manifest",
                description="Use this when you want one audit table for uploads, staged downloads, copied-back outputs, and finalized local states.",
                rows=manifest_rows,
                key_prefix="operations_transfer_manifest",
            )
            staging_rows = [row for row in manifest_rows if str(row.get("staging_path", "-")) not in {"", "-"}]
            if staging_rows:
                st.markdown("#### Staging paths in use")
                st.dataframe(
                    [
                        {
                            "cluster": row.get("cluster", ""),
                            "target": row.get("target", ""),
                            "staging_path": row.get("staging_path", ""),
                            "next_action": row.get("next_action", ""),
                        }
                        for row in staging_rows
                    ]
                )
            st.markdown("#### Artifact verification")
            if artifact_attention:
                st.caption("These rows are the transfer-adjacent integrity blockers: missing staged downloads, incomplete copied-back outputs, or file mismatches that make the remote/local handoff unsafe to trust.")
            _render_artifact_verification_workspace(
                st,
                artifact_verification,
                title="Artifact integrity checks",
                caption="This verifies whether the expected files really exist for the visible transfers and campaigns: staging paths, runtime outputs, SASA/AP summaries, and ingest files.",
                key_prefix="operations_transfers",
                render_export_pack=_render_export_pack,
            )
            st.markdown("#### Raw sync records")
            st.dataframe(sync_records)
        else:
            st.info("No sync records have been captured yet.")
        return

    st.subheader("Run curation")
    st.caption("Pin the runs that represent real thesis work, hide noisy smoke / bugcheck / tuning runs, and optionally give important runs a clearer label. Nothing here deletes any run folder.")
    all_runs = list(state.get("all_runs", state.get("runs", [])))
    if not all_runs:
        st.info("No runs found to curate.")
        return
    for run in all_runs:
        run_key = str(run.get("run_dir", ""))
        row_cols = st.columns([2.8, 1, 1, 1.6])
        row_cols[0].markdown(f"**{run.get('run_display_name', run.get('run_name', ''))}**")
        row_cols[0].caption(run.get("run_identity", ""))
        row_cols[0].caption(
            f"Workspace role: {'Historical / Test' if run.get('is_historical_candidate') else 'Current Thesis candidate'} | Pinned: {_friendly_bool(bool(run.get('is_pinned')))} | Hidden: {_friendly_bool(bool(run.get('is_hidden')))}"
        )
        pin_key = f"pin_run_{run.get('run_slug', '')}"
        hide_key = f"hide_run_{run.get('run_slug', '')}"
        label_key = f"label_run_{run.get('run_slug', '')}"
        if run.get("is_pinned"):
            if row_cols[1].button("Unpin", key=pin_key):
                unpin_dashboard_run(Path(str(state["run_root"])), run_key)
                _stash_dashboard_flash(st, "success", f"Unpinned {run.get('run_display_name', run.get('run_name', 'run'))}.")
                _trigger_dashboard_rerun(st)
        else:
            if row_cols[1].button("Pin", key=pin_key):
                pin_dashboard_run(Path(str(state["run_root"])), run_key)
                _stash_dashboard_flash(st, "success", f"Pinned {run.get('run_display_name', run.get('run_name', 'run'))} for Current Thesis Work.")
                _trigger_dashboard_rerun(st)
        if run.get("is_hidden"):
            if row_cols[2].button("Show", key=hide_key):
                show_dashboard_run(Path(str(state["run_root"])), run_key)
                _stash_dashboard_flash(st, "success", f"Restored {run.get('run_display_name', run.get('run_name', 'run'))} to visible workspaces.")
                _trigger_dashboard_rerun(st)
        else:
            if row_cols[2].button("Hide", key=hide_key):
                hide_dashboard_run(Path(str(state["run_root"])), run_key)
                _stash_dashboard_flash(st, "success", f"Hid {run.get('run_display_name', run.get('run_name', 'run'))} from the default workspace.")
                _trigger_dashboard_rerun(st)
        new_label = row_cols[3].text_input(
            "Run label",
            value=str(run.get("user_label", "")),
            key=label_key,
            help="Optional user-facing label shown alongside the configured run name and folder slug.",
        )
        if row_cols[3].button("Save label", key=f"save_{label_key}"):
            set_dashboard_run_label(Path(str(state["run_root"])), run_key, new_label)
            _stash_dashboard_flash(st, "success", f"Updated label for {run.get('run_display_name', run.get('run_name', 'run'))}.")
            _trigger_dashboard_rerun(st)
        st.divider()
