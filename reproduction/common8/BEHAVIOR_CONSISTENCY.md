# Actual sampling denominator for common8

The previous native discovery (3278, runtime `130d17a0`) stopped after one
Adam write per rank and zero completed outer updates. Its old probabilities
were recomputed with HF on the final assembled trajectory. They were not a
receipt for the distribution that produced each action. Its post-write KL
estimate of 0.8070046 therefore does not establish actual behavior-policy KL.
The failed source, logs and probes remain separate and immutable.

## Separate implementation changes

This common8 path carries each real generation call's request IDs, raw output
IDs, selected token log-probabilities, finish/stop reason, sampler settings,
trajectory identity and policy snapshot through the controller. Tensor and
non-tensor row identities are padded, trimmed and reordered together.

The actual raw token prefix follows the original boundary: retain through the
first search closing tag if present, otherwise the first answer closing tag;
model output is never decoded and then re-tokenized into a different action.
A closing character boundary may fall inside a BPE token. The complete
crossing token and its actual selected logprob and actor mask are retained,
including any suffix. The executor receives the original character prefix;
subsequent requests, actor/reference recomputation and strict authored-span
reward decoding retain the complete token. New proofs record the exact suffix
and classify it as absent, whitespace or nonwhitespace; original whitespace-only
proofs remain readable with their original strict meaning. A nonwhitespace
suffix is never labelled whitespace or removed to make an answer valid.

This token-atomic adaptation preserves the original action projection, not the
old retokenized next-request context or each trajectory's old reward. A terminal
answer with nonwhitespace suffix remains invalid under the existing strict QA
parser. No token is split, assigned a fabricated partial-token probability,
retokenized, masked out or used as a cleaned reward span. Native discovery3288
with immutable source3fdf stopped on the previous extra whitespace-only guard;
the new protocol is tested with synthetic CPU token fragments, not claimed as
an observation of that job's actual tokenizer or as a passed GPU gate.

Decoding is a deterministic text projection and may be lossy. A genuine
replacement character is retained verbatim in executor queries, context and
strict reward text. New closed proofs carry the decoded full generation, a
projection role explicitly disclaiming UTF-8 validity, and exact full/retained
replacement-character counts. This does not prove valid raw UTF-8 or distinguish
an encoded literal replacement from invalid bytes. Full text must start with
the retained decoded prefix exactly; no-tag projections must equal it. The
prefix is never extended until Unicode improves. Older proof schemas preserve
their original no-replacement meaning.

Native3288's separate malformed-byte row was checked privately against the
original tokenizer decoder; the original and explicit decoder projections and
search action were equal with a stable prefix. The new CPU tests use synthetic
decoder projections and do not stand in for native probability acceptance.

Ambiguous special tokens, unstable Unicode prefix boundaries, altered prefixes and
context-window overflow reject the complete outer, with private evidence.
No trajectory is discarded to make the gate pass.

Runtime `a09a5530` incorrectly rejected an answer closing tag followed by a
search closing tag as “noncausal”; native discovery3283 stopped during the
initial paired512 evaluation, before a behavior gate or completed outer.
Independent execution of the original parser/executor showed that this was an
extra implementation bug. The old validation-failure handler returned without
writing a private raw journal, so the stopped3283 trace cannot be reconstructed
or retroactively claimed. Future evaluation failures use an independent private
validation journal with run/job/source identity, snapshot nonce and failed-only
scope, including logical step0. They do not replace or certify training outers.
Upstream truncation gives search closing precedence,
then the executor selects the first complete action. An answer followed by a
search therefore keeps both sampled spans but executes answer/done with no
retrieval. The correction retains all original IDs, selected probabilities and
actor masks through that boundary, without moving or cleaning either span.
Strict QA extraction still rejects the mixed authored span: EM/F1 stay zero,
and the active Search-G1 invalid-answer reward floor remains -0.1. This does
not establish native probability parity or a healthy policy.

The initial request and final HF prompt both select the original last 2048
prompt tokens. Retrieval observations and correction feedback remain context
and are excluded from the actor objective. Their probabilities are zero in
the frozen old tensor. Private journals retain actual IDs and are written
only in the native artifacts directory; published evidence contains hashes
and scalars.

Sampling `top_p` changes explicitly from 0.95 to 1.0, with temperature 1 and
neutral filters/penalties. This is a separate sampling correction: it enables
the selected full-softmax probability and the HF denominator to refer to the
same distribution. It is not an effect attributed to Dual-Clip. Raw-token
retention can also change the retained token counts and numerical masked-mean
denominators compared with the former re-tokenization path.

The official model/revision, original data bytes/order, 512 questions × 5,
mini128, epoch1, 20 Adam writes per rank, global actor micro16, old/ref micro32,
learning rate1e-6, reference KL, grouped advantages, G1 reward, action budget
and original two-row masked-mean reduction formula remain unchanged.

## Admission and receipt scope

Before any old/reference forward, audit all 2560 actual trajectories against
their final actor input, position IDs and masks. Then independently recompute
HF probabilities with local micro4 and require absolute error ≤0.05 on every
retained actor token. Freeze the passed actual selected probabilities as old.
The full gate precedes Adam; it does not use the greedy model smoke test as
substitute evidence.

Each worker collectively verifies its 320 row bytes, original group IDs and
the global 2560 unique-trajectory coverage before a local assertion or write.
The gate SHA links pre/post monitoring, actual Adam writes, completed-update
records, five-step evidence and final verification. Historical three-field
contexts remain readable; new contexts must carry one consistent gate SHA
through all ranks and checks. Lightweight receipt readers authenticate these
scalar/hash links, not the raw journal or a new GPU execution.

The existing KL brake still checks a fixed 32-trajectory length-stratified
subset before and after each mini-batch, using a globally weighted k3 mean.
Its paired two-row weights use the original reduction operator but need not
equal each selected token's denominator in its original training micro-batch.
It is an empirical subset statistic, not a full-rollout or unbiased population
KL guarantee. Candidate KL0.01, Dual-Clip3/epsilon0.2 and numerical bound30
retain their candidate status. A post-write rejection does not undo that write.

Local CPU protocol tests, target CPU/Gloo regressions, greedy model smoke,
native all-token behavior parity, a complete 20-write discovery outer and an
independent confirmation are distinct requirements. This source change alone
proves none of the native gates or healthy-policy admission. Formal training
remains zero until the new source passes those actual gates.
