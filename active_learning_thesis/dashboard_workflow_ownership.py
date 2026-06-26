from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkflowOwner:
    action_kind: str
    view: str
    section: str
    title: str
    summary: str


_OWNERS: tuple[WorkflowOwner, ...] = (
    WorkflowOwner(
        "init-run",
        "Operations",
        "New run wizard",
        "Create run",
        "New thesis runs are created only from Operations -> New run wizard.",
    ),
    WorkflowOwner(
        "run-study",
        "Operations",
        "Study designer",
        "Run or dry-run study",
        "Multi-seed study planning, execution, summary, and comparison live in Operations -> Study designer.",
    ),
    WorkflowOwner(
        "summarize-study",
        "Operations",
        "Study designer",
        "Summarize study",
        "Study summaries are created from Operations -> Study designer, then viewed in Results.",
    ),
    WorkflowOwner(
        "compare-studies",
        "Operations",
        "Study designer",
        "Compare studies",
        "Study comparison jobs are launched from Operations -> Study designer and viewed in Results.",
    ),
    WorkflowOwner(
        "supek-submit-study",
        "Operations",
        "Study designer",
        "Submit SUPEK study",
        "SUPEK multi-seed study submission belongs in Operations -> Study designer.",
    ),
    WorkflowOwner(
        "supek-submit-study-array",
        "Operations",
        "Study designer",
        "Submit SUPEK study array",
        "Per-seed SUPEK array submission belongs in Operations -> Study designer.",
    ),
    WorkflowOwner(
        "continue-al-feedback",
        "Model Workflow",
        "Local model actions",
        "Continue AL from feedback",
        "Local model-loop actions are launched from Model Workflow -> Local model actions.",
    ),
    WorkflowOwner(
        "ingest-round",
        "Model Workflow",
        "Local model actions",
        "Ingest returned labels",
        "Model label-ingest actions are launched from Model Workflow -> Local model actions.",
    ),
    WorkflowOwner(
        "run-replay",
        "Model Workflow",
        "Local model actions",
        "Run replay benchmark",
        "Replay and local model workflow commands are launched from Model Workflow -> Local model actions.",
    ),
    WorkflowOwner(
        "propose-round",
        "Model Workflow",
        "Local model actions",
        "Propose next batch",
        "Model proposal commands are launched from Model Workflow -> Local model actions.",
    ),
    WorkflowOwner(
        "freeze-final",
        "Model Workflow",
        "Thesis freeze",
        "Freeze final model state",
        "Thesis freeze actions are launched from Model Workflow -> Thesis freeze.",
    ),
    WorkflowOwner(
        "supek-verify-env",
        "Model Workflow",
        "Remote SUPEK",
        "Verify SUPEK environment",
        "SUPEK model workflow controls live in Model Workflow -> Remote SUPEK.",
    ),
    WorkflowOwner(
        "supek-sync-repo",
        "Model Workflow",
        "Remote SUPEK",
        "Sync SUPEK repo",
        "SUPEK model workflow controls live in Model Workflow -> Remote SUPEK.",
    ),
    WorkflowOwner(
        "supek-sync-run",
        "Model Workflow",
        "Remote SUPEK",
        "Upload run state to SUPEK",
        "SUPEK model workflow controls live in Model Workflow -> Remote SUPEK.",
    ),
    WorkflowOwner(
        "supek-preflight",
        "Model Workflow",
        "Remote SUPEK",
        "Run SUPEK preflight",
        "SUPEK model workflow controls live in Model Workflow -> Remote SUPEK.",
    ),
    WorkflowOwner(
        "supek-submit-workflow",
        "Model Workflow",
        "Remote SUPEK",
        "Submit SUPEK workflow",
        "SUPEK model workflow controls live in Model Workflow -> Remote SUPEK.",
    ),
    WorkflowOwner(
        "supek-poll-qstat",
        "Model Workflow",
        "Remote SUPEK",
        "Poll SUPEK queue",
        "SUPEK model workflow controls live in Model Workflow -> Remote SUPEK.",
    ),
    WorkflowOwner(
        "supek-pull-artifacts",
        "Model Workflow",
        "Remote SUPEK",
        "Pull SUPEK artifacts",
        "SUPEK model workflow controls live in Model Workflow -> Remote SUPEK.",
    ),
    WorkflowOwner(
        "supek-fetch-logs",
        "Model Workflow",
        "Remote SUPEK",
        "Fetch SUPEK logs",
        "SUPEK model workflow controls live in Model Workflow -> Remote SUPEK.",
    ),
    WorkflowOwner(
        "supek-cancel-job",
        "Model Workflow",
        "Remote SUPEK",
        "Cancel SUPEK job",
        "SUPEK cancellation remains in Model Workflow -> Remote SUPEK and still requires approval.",
    ),
    WorkflowOwner(
        "prepare-manual-md-stage",
        "MD Validation",
        "Manual MD sandbox",
        "Prepare manual MD sandbox",
        "Manual peptide MD campaigns are prepared from MD Validation -> Manual MD sandbox.",
    ),
    WorkflowOwner(
        "prepare-md-stage",
        "MD Validation",
        "Local MD actions",
        "Prepare AL MD stage",
        "AL-driven MD preparation is launched from MD Validation -> Local MD actions.",
    ),
    WorkflowOwner(
        "finalize-md-stage",
        "MD Validation",
        "Review & ingest",
        "Parse returned MD stage",
        "Returned MD parsing, review, and ingest handoff live in MD Validation -> Review & ingest.",
    ),
    WorkflowOwner(
        "bura-md-workflow",
        "MD Validation",
        "Remote BURA",
        "Run full BURA MD workflow",
        "One-button Dash/BURA MD workflow queues local preparation plus BURA autopilot from MD Validation -> Remote BURA.",
    ),
    WorkflowOwner(
        "bura-upload-campaign",
        "MD Validation",
        "Remote BURA",
        "Upload campaign to BURA",
        "BURA upload, preflight, submit, poll, pull, and logs live in MD Validation -> Remote BURA.",
    ),
    WorkflowOwner(
        "bura-normalize-scripts",
        "MD Validation",
        "Remote BURA",
        "Normalize BURA scripts",
        "BURA upload, preflight, submit, poll, pull, and logs live in MD Validation -> Remote BURA.",
    ),
    WorkflowOwner(
        "bura-preflight",
        "MD Validation",
        "Remote BURA",
        "Run BURA preflight",
        "BURA upload, preflight, submit, poll, pull, and logs live in MD Validation -> Remote BURA.",
    ),
    WorkflowOwner(
        "bura-submit-chain",
        "MD Validation",
        "Remote BURA",
        "Submit BURA chain",
        "BURA upload, preflight, submit, poll, pull, and logs live in MD Validation -> Remote BURA.",
    ),
    WorkflowOwner(
        "bura-poll-squeue",
        "MD Validation",
        "Remote BURA",
        "Poll BURA queue",
        "BURA upload, preflight, submit, poll, pull, and logs live in MD Validation -> Remote BURA.",
    ),
    WorkflowOwner(
        "bura-pull-package",
        "MD Validation",
        "Remote BURA",
        "Pull BURA package",
        "BURA upload, preflight, submit, poll, pull, and logs live in MD Validation -> Remote BURA.",
    ),
    WorkflowOwner(
        "bura-fetch-logs",
        "MD Validation",
        "Remote BURA",
        "Fetch BURA logs",
        "BURA upload, preflight, submit, poll, pull, and logs live in MD Validation -> Remote BURA.",
    ),
    WorkflowOwner(
        "bura-cancel-job",
        "MD Validation",
        "Remote BURA",
        "Cancel BURA job",
        "BURA cancellation remains in MD Validation -> Remote BURA and still requires approval.",
    ),
    WorkflowOwner(
        "update-md-review",
        "MD Validation",
        "Review & ingest",
        "Review MD label",
        "Human MD label review belongs in MD Validation -> Review & ingest.",
    ),
    WorkflowOwner(
        "make-md-ingest-csv",
        "MD Validation",
        "Review & ingest",
        "Create MD ingest CSV",
        "Returned-label ingest handoff belongs in MD Validation -> Review & ingest.",
    ),
    WorkflowOwner(
        "export-md-source-batch",
        "Peptides",
        "Candidate selection",
        "Choose MD candidates",
        "Candidate choice belongs in Peptides -> Candidate selection.",
    ),
    WorkflowOwner(
        "export-thesis-packet",
        "Results",
        "Thesis output builder",
        "Export thesis packet",
        "Thesis packet, figure, canary, and export actions belong in Results -> Thesis output builder.",
    ),
    WorkflowOwner(
        "build-thesis-figures",
        "Results",
        "Thesis output builder",
        "Build thesis figures",
        "Thesis packet, figure, canary, and export actions belong in Results -> Thesis output builder.",
    ),
    WorkflowOwner(
        "thesis-canary",
        "Results",
        "Thesis output builder",
        "Run thesis canary",
        "Thesis packet, figure, canary, and export actions belong in Results -> Thesis output builder.",
    ),
)

OWNERS_BY_ACTION_KIND = {owner.action_kind: owner for owner in _OWNERS}


def canonical_owner(action_kind: str) -> WorkflowOwner | None:
    return OWNERS_BY_ACTION_KIND.get(str(action_kind))


def is_canonical_context(action_kind: str, view: str, section: str) -> bool:
    owner = canonical_owner(action_kind)
    if owner is None:
        return True
    return owner.view == str(view) and owner.section == str(section)


def canonical_navigation_hint(action_kind: str) -> dict[str, str]:
    owner = canonical_owner(action_kind)
    if owner is None:
        return {
            "view": "",
            "section": "",
            "title": "Primary page unknown",
            "summary": "This action has no workflow-owner metadata yet.",
        }
    return {
        "view": owner.view,
        "section": owner.section,
        "title": owner.title,
        "summary": owner.summary,
    }


def view_section_session_key(view: str) -> str:
    return {
        "Model Workflow": "dashboard_model_section",
        "MD Validation": "dashboard_md_section",
        "Operations": "dashboard_operations_section",
        "Peptides": "dashboard_peptides_section",
        "Results": "dashboard_results_section",
    }.get(str(view), "")


def view_section_query_key(view: str) -> str:
    return {
        "Model Workflow": "model_section",
        "MD Validation": "md_section",
        "Operations": "operations_section",
        "Peptides": "peptides_section",
        "Results": "results_section",
    }.get(str(view), "")
