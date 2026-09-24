# Streamlit UI Contract

The UI displays the current stage, run status, todo progress, search-call budget, latest supervisor
assessment, verification result/issue, recovery attempt, artifact/evidence links, and approval
controls. It submits chat turns and approval decisions to the graph runner only.

The UI must show: clarification questions in Scoping; automatic transition to Planning; visible
ResearchComplete status; `NEEDS_REVIEW` with unresolved feedback; resume errors with the latest valid
checkpoint option; and publication success/failure. It must never present an unverified artifact as
verified or expose a publication control for an ineligible artifact.
