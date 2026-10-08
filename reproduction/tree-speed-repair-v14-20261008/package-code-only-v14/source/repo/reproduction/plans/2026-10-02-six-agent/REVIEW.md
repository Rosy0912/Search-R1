# Independent plan review

Status: Approved (2026-10-02).
Reviewer: review_six_agent_plan; read-only checks against run.sh, config, preflight, holdout and environment/actor checkpoint code.

Resolved findings:
1. Per-agent absolute paths cause different holdout manifest IDs: compare path-independent source/parameter/question/row content identity and output SHA; preserve path provenance separately.
2. No existing holdout checkpoint selector: require a reviewed new holdout-only evaluator, candidates actual update5/100/200/300/400/500, maximize mean answerF1,ties earliest,never final test; use complete saved actor HF directory.
3. TOP-D needs trained teacher even for smoke: CPU/general hardware preparation can precede teacher; algorithm smoke/acceptance/formal train cannot.

Six independent8GPUH100 allocations and dual15% rules checked. Existing low-memory processes preserved. Preflight process-policy update, portable asset paths,CPU retrieval configuration and selector are pending implementation prerequisites, not falsely claimed ready. No GPU training or reservation performed.
