from __future__ import annotations

from active_learning_thesis.dashboard_action_contracts import get_dashboard_action_contract


def _joined(values: tuple[str, ...] | list[str]) -> str:
    clean = [str(value).strip() for value in values if str(value).strip()]
    return "; ".join(clean) if clean else "None"


def render_action_contract_compact(st, contract_id: str, readiness: dict[str, object] | None = None) -> None:
    contract = get_dashboard_action_contract(contract_id)
    if contract is None:
        return
    if readiness:
        verdict = str(readiness.get("verdict", "")).strip()
        summary = str(readiness.get("summary", "")).strip()
        blockers = [str(item).strip() for item in list(readiness.get("blockers", [])) if str(item).strip()]
        cautions = [str(item).strip() for item in list(readiness.get("cautions", [])) if str(item).strip()]
        if verdict or summary:
            st.caption("Readiness: " + " | ".join(item for item in [verdict, summary] if item))
        if blockers:
            st.warning("Blocked by: " + "; ".join(blockers[:4]))
        elif cautions:
            st.info("Heads-up: " + "; ".join(cautions[:4]))
        fix_now = str(readiness.get("fix_now", "")).strip()
        if fix_now:
            st.caption(f"Fix blocker: {fix_now}")

    st.markdown(f"**Prerequisites:** {_joined(contract.prerequisites)}")
    st.markdown(f"**Postcondition:** {contract.postcondition}")
    approval_label = {
        "required": "Approval required",
        "not_required": "Runs immediately",
        "not_applicable": "No dashboard action file is created",
    }.get(contract.approval, contract.approval)
    st.caption(f"Action contract: {approval_label} | scope={contract.scope} | cluster={contract.cluster or 'local'}")

    with st.expander("Advanced action contract details", expanded=False):
        st.markdown(f"**Writes:** {_joined(contract.writes)}")
        st.markdown(f"**Safer option:** {contract.safer_option}")
        st.markdown(f"**Recovery:** {contract.recovery}")
        st.markdown(f"**Metadata keys:** {_joined(contract.metadata_keys)}")
        st.markdown(f"**Output suffixes:** {_joined(contract.output_suffixes)}")
        st.markdown(f"**Action kind:** {contract.action_kind or 'none'}")

