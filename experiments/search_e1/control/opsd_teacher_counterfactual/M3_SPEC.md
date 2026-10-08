# M3 fixed-weight teacher counterfactual specification

2026-10-08. Design only: no implementation, model load, GPU, HTTP, deployment,
training or Git operation is authorized by this document. Root controls subsequent
review/publication/admission. No private question, answer or response body appears
here. This follows the approved degradation plan's M3, with M4 GPU priority.

## 1. Scientific question and fixed objects

H1 has two distinct claims: the teacher may depend on privileged reference content,
and it may fail to repair an ordinary student's failed history. Reference-sensitive
distributions support conditional dependence; better privileged continuation
weakens repair failure but does not disprove privilege mismatch. Fixed-weight
interventions do not identify the training effect of removing references.

Use the frozen source100 base, BF16, HF SDPA, exact tokenizer/chat template,
generation_config and enabled processor stack. Teacher adapter stays disabled,
eval/no_grad throughout; no optimizer/backward/weight save. Ordinary source100
with its verified zero-B adapter is an additional identity/distribution control,
not another independently trained teacher. Verify its no-reference behavior
against adapter-disabled no-reference base; report actual dtypes and tolerances.
Do not use update1/update5/epoch1 as the privileged teacher.

Four reference conditions, with recipient question/history/state fixed:

| Condition | Exact prompt intervention | Interpretation |
|---|---|---|
| original | `core.prompt(question, original_successful_reference)` | Original training privilege |
| none | `core.prompt(question, None)` | Omit the entire reference envelope, not an empty reference string |
| remove_final_answer | Delete only the final complete authored answer tag plus its payload from the original reference; keep all other bytes | Final-answer-block deletion, not removal of all answer information |
| mismatched | A precommitted other-question successful reference, unedited | Cross-question reference intervention, not an expert or valid reference for recipient |

The unchanged envelope is the actual core lines12–17: instruction/question, then
`Expert reference trajectory`, the reference, and the request to attempt the same
problem in the student's own words. Answers are used for frozen scoring and donor
checks only, never separately appended to teacher/student prompt. Reference text
itself receives no response loss. Successful reference means accepted final action,
not verified reasoning or fully balanced transcript.

Deletion locates the native last authored span on the exact chosen reference row,
verifies frozen QA accepts its final answer and deletes that complete answer-tag
character interval only. Earlier policy/information mentions remain. Resolve the
reference sibling using original source order and the native min key; do not infer
unique origin merely from identical response text. Ambiguous source matches remain
reported. If native-selected reference/span cannot be established, fail preparation.

Within each of the four selected dataset/group cells, sort preselected cases by
the same fixed selection key and choose donor via deterministic cyclic search over
other cases. Prefer the first donor whose accepted final answer does not match any
recipient gold under frozen normalization, without observing counterfactual output.
Record donor question/reference identities and rule. If no such donor exists,
explicitly skip that condition rather than call a coincidentally matching answer
false. Cross-question mismatch is still not proof that every donor statement is
factually false. Content, length, positions and repetition-processor history all
change together; interpret effects as whole-prompt interventions.

## 2. Sources actually inspected and current evidence

Frozen runtime source identities:

- `search_e1/core.py`: `0fbfbe63d0aafb41f79db702e08c9a0efc5f9bbf29a2964310594ec2ec0a83fc`.
- `search_e1/opsd.py`: `61df9cc630bca7466f3edfb06da18ad92ddff1c713bf7c6c4a104ba88672c037`.
- `search_e1/rollout.py`: `a88506797ea7fe24e61e1ad53ba4487aef87bdd69ec395aef26c94973b3d629a`.
- `search_e1/vllm_rollout.py`: `acddb14f00661fdc3f928ef58e6bef3a0125c34b5582f8f9441cd7e9d7362859`.
- Frozen QA: `9a825a7b222b5a3fd2c1de7f63ed38e157258fcf858e773ff64c963a4b6daddd`.
- Existing diagnostic `trace_agent.py`: `d2e0ea7eff28cdc6a4ce4fd5ff1f81d323216ab8c28d3052348a86388255d906`.

Source references: native `rollout.py` lines24–67 initializes all state internally;
lines29–38 build context and generation kwargs; lines41–53 decode, cut and execute
terminal/search transitions; lines54–66 retrieve and insert observations. Existing
`trace_agent.py` lines29–86 records the same HF steps, including exact context,
actual generated IDs, structural cut, original retrieval payload and inserted text.
OPSD `encoded_pair`/`pair_loss` separately encode response and prompt, use causal
prefix-length-minus-one alignment, disabled-adapter frozen q and policy masks.

Only three already-local source100 traces were inspected; no new inference was
requested. For each, reconstructing prompt + previous policy/observation + forced
text matched every recorded context_text; policy spans and inserted_information
reconstructed final response and calls exactly. All three had two turns and one
search. Their artifact SHA256 values are:

- `8739a34a6c9788f08ceca1131b94c3eb64e20dab3ba3396817289d7521a457d0`.
- `27b201d44edd16749810309109c7d07aa79ae02ba959e4cf3172db60414a643a`.
- `50d804a7bf3fd1e6b6521682ab8080271dbb0e15966472bbb5d254c52aae8bd6`.

This validates that those real trace fields support state reconstruction. It does
not validate all192 cases, original vLLM generated IDs, forced-turn examples or
any new continuation probability/performance.

## 3. Approach choice

1. **Recommended: an independent explicit-state HF executor wrapper.** New
   `reconstruct_state` is CPU-only; new `continue_state` reproduces native HF
   transitions with an injected initial state and reference prompt. Keep native
   StopTags, core prompt/reward, model.generate options and observation formatting.
   Whole-run CPU stubs compare against the unmodified frozen SearchAgent.
2. Call native `SearchAgent.run` with a fabricated question/history: rejected.
   It changes the user prompt, resets response/calls, and grants five new turns.
3. Intercept native generation to replay all prior turns before the boundary:
   possible but unnecessarily fragile (retrieval replays, timing/context accounting,
   first-turn processors and adapter modes). Do not use this for the primary design.

The wrapper is an explicitly new evaluation executor, not a silent patch to
training or native512. Native fresh evaluation remains the frozen run path.

## 4. Canonical state and two boundary definitions

Canonical state records `case/question/reference/source-row` hashes, original
backend, boundary ID, original next turn t, response_before_turn, prior authored
policy spans, calls_completed, previous observations, terminal state and its cause.
Turn t is zero-based; remaining generation turns are `5-t`; remaining searches
are `4-calls_completed`. State must be awaiting **generation**, never awaiting
retrieval. Record original response-prefix SHA and character extent; private text
stays only in private artifacts. Past turns and observations remain byte-identical
in all conditions; later branch observations may diverge and are recorded.

The imminent forced text is a pending transition, not normally stored inside
response_before_turn: it is the exact frozen literal iff t=4. A separate record
states whether the saved source already includes that literal. Normalize to the
pre-turn response once, preserving original-byte provenance, and insert once when
constructing generation context. Never derive it by deleting an arbitrary matching
information block; its exact gap position/source-turn evidence must establish it.

Boundary A (`after_first_policy_and_observation`): execute the first original
policy transition. If it is a search that actually retrieved, include its entire
already-recorded observation and set t=1/calls=1. If it ends answer, invalid or
search budget, skip this boundary; do not resume after a terminal first action.
An unfinished/missing observation is a preparation error/explicit unsupported
state, not an invitation to repeat the original HTTP request.

Boundary B (`before_last_policy_generation`): the original executable state just
before its last emitted policy span. Include all preceding complete observations,
set t to that span's original turn and preserve pending forced text. This is not
an arbitrary token immediately before an answer tag. The last policy may later
produce an invalid answer or another terminal action; that future result does not
make its earlier pre-generation state terminal. Label whether the original last
policy actually contained an answer. An actual context-limit attempt with no
generation is terminal/skipped; do not pretend a previous policy is that attempt.

If A and B identify the same state (common in two-turn search→answer traces),
run that state once per condition, with both boundary aliases. Keep the two target
estimands' denominators explicit; duplicate aliases are not independent samples.
For a one-turn final response, A skips and B is the original t=0 empty history;
label `no_failure_history`, not a repaired failed prefix. For original CC pairs,
mark history from a successful student rather than calling it failed. CW mined
students are original failures. Hard groups have no successful teacher and remain
outside these four reference conditions.

Skip only if the **selected boundary state** is already terminal or unsupported.
A later original terminal result must not retroactively suppress an earlier valid
search boundary. Conversely, never continue a state after terminal answer,
invalid_or_budget, context_limit or exhausted-loop budget. No repaired tags,
truncation, hidden new observations or extra search/turn opportunities.

## 5. Reconstructing state from two different evidence types

### Real HF debug traces (strong transition evidence)

Replay event metadata without inference/HTTP. Verify turn indices contiguous,
context_text = original prompt_text + accumulated response + forced literal,
policy_segment = response[policy_span], and its decoded-before-cut segment after
the original earliest closing-search/answer cut equals policy_segment. Append
forced text exactly where the native HF executor did, then the policy, then
inserted_information iff a retrieval occurred. Require query, reply, document
formatting/480-token insertion metadata and search_calls to reconcile. A skipped
context-limit event has no generated IDs/policy span and appends no forced text
under HF semantics. Tokenization checks are a later tokenizer-only stage using
the approved tokenizer, not a CPU speculative character count.

At a pre-generation boundary, reconstruction must match that event's exact
context_text for original/no-reference source100 mode. Altering reference then
rebuilds only prompt prefix, while preserving the same accumulated response and
state. New teacher own-history traces are not substituted for student histories.

### Preserved mined original trajectories (primary training-history evidence)

These originate from the preserved vLLM rollout file, with response, policy_spans,
search_calls and termination but no per-turn debug events or generated IDs.
Each policy span corresponds to one emitted turn in original order. Replay the
native action transition from each exact span: closing answer is terminal before
search regex; otherwise find the same first search regex and enforce original
call limit. Earlier malformed think is not grounds for terminating a search that
the actual executor accepted; frozen final-action scoring and executor decisions
are different rules.

Between consecutive policy spans, require the exact original external observation
gap, and on the last turn the exact forced suffix if present. Preserve the gap
bytes verbatim; never call the retriever to reconstruct them. Validate gaps against
the original vLLM insertion wrapper and no unaccounted bytes. Document content may
itself contain tag-like strings, so parse gap extent from authored span provenance
and known final forced suffix, not a naive first closing-information regex. Missing
provenance/ambiguous forced gap is unsupported and explicitly skipped. Validate
final calls/termination and distinguish actual last generated policy from an
un-emitted context-limit attempt. Use source line/row hash and native-selected
sibling identity to bind the reconstructed history.

Important source difference: original vLLM appends forced information to response
**before** the turn4 context check. HF builds `text` with forced but appends it to
response **after** successful generate. Thus a vLLM turn4 context-limit artifact
may end with an un-authored forced gap while the HF counterpart does not. Record
this original backend distinction and normalize only valid active boundaries; do
not claim the old vLLM trajectory would be regenerated by the HF wrapper. Original
vLLM IDs/probabilities are unavailable; never manufacture them by retokenization.

## 6. Continuation transition and protocol

For each condition, derive prefix using frozen core.prompt and actual chat
template. Starting at original t, repeat only t…4. Construct text from condition
prefix + current response_before_turn + exact pending forced text. Tokenize that
whole string with add_special_tokens=False, and reject context when
`input_tokens + 500 > 8192`, exactly as native (8192 alone is not the admission
threshold). No independent reference/history truncation. Context skips are a
condition-dependent intervention effect; report common-condition sample set plus
all individual skips, without pretending matched lengths or selecting only wins.

Use max_new_tokens=500, do_sample=False, native EOS pad and native StopTags start
at actual input length. Preserve inherited generation_config (including observed
repetition_penalty1.05) and actual processor stack rather than imposing an invented
temperature-only configuration. Decode actual generated IDs with skip_special_tokens
and retain raw decoded text before cutting at earliest closing search/answer.
Append forced, policy segment and authored span exactly as native. Closing answer
terminates even if strict QA later rejects it. No search match or calls=4 gives
invalid_or_budget. Otherwise use original top3/return_scores=True request, actual
timeout/health identity checks, original document formatting and 480 content-token
limit, complete information wrapper, calls+1. If loop finishes, report budget.

Report native frozen reward on full resumed response with preserved+new authored
spans, plus native final-action protocol validity, search counts and termination.
Past information remains context/masked. Literal answers in information/earlier
policy or unclosed final think remain excluded from strict EM. Neither valid-gold
mismatch nor structural-category labels claim factual/semantic error.

Historical observations are reused only up to the fork; new branch queries use
the admitted resident retriever. Distinguish prerecorded observation replay from
new HTTP responses. Actual model/tokenizer/source/GPU/retriever gates and lock
remain Root-owned, with M1 completion and M4 GPU priority; this spec adds no launcher.

## 7. Three probability/loss objects that must remain separate

1. **Native training-history q:** for original response y, use frozen encoded_pair
   conventions: independently encode prompt and y, concatenate those IDs, use
   prefix-length−1+i predicting y_i, frozen adapter-disabled teacher raw logits at
   implicit temperature1, original policy mask and original valid chunk32. Original
   reference/none results already recorded are reused. New reference conditions
   require separate fixed-history probes later. Native joint 8192 response+prompt
   admission applies to these probes, not the generation +500 rule. No CE on
   failed student labels is silently introduced; original objective is capped
   per-vocabulary teacher→student forward KL, then valid-token/trajectory means.
2. **Executable-boundary distribution:** tokenize condition prompt plus exact
   complete executor history jointly, as native HF generate does. Save actual
   input IDs/raw logits plus HF-processed scores and relevant action-token ranks.
   Independently encoded prompt/history can tokenize differently at joins, so
   this distribution is not automatically identical to object1. Repetition penalty
   sees the condition's full input and generated history; its change is part of
   prompt intervention, not evidence that raw training q changed for that reason.
3. **Actual generated continuation:** actual generated IDs/decoded-before-cut/
   kept segment, policy spans, queries, documents, termination and strict EM. A
   later tokenizer pass on kept response yields posthoc IDs and forcing loss, not
   original online generation probabilities. Preserve actual vs posthoc separately.

Same-history native q_priv must be invariant across student adapter checkpoints;
existing M2 comparison checks observed target/top32/logp/entropy/IDs with explicit
missing fields, not full-vocabulary reconstruction. Teacher's own autonomous
rollout uses its own histories and cannot demonstrate every training token's q
quality. Likewise a teacher that completes one failed prefix successfully does
not prove original full-response KL targets were correct everywhere.

M3 cross-history loss supplement: reuse source100/epoch1 actual responses already
generated by M1; score each fixed response under all four student checkpoints
with the original successful reference/teacher. Assign history-owner/version and
trace SHA explicitly. It is new-response posthoc forcing, not original training
loss; same response is compared across checkpoints, rather than comparing their
different own-history means. No new source100/epoch1 rollout is needed for this
cross-scoring where existing accepted artifacts suffice.

## 8. Fixed64, pilot16 and output contract

After M1 case identities are accepted, select16 from each of the four successful-
reference cells using precommitted seed42/hash ordering, save64 manifest before
observing counterfactual outcomes. Pilot uses first4 in each cell (16); verify
reconstruction/base control/capacity/records, then continue same run on remaining48
without repeating pilot or stopping on favorable results. Seen regression examples
plus matched stable/improved cases stay separately labeled case-study, not mixed
into64 effect proportions. All5wrong lacks teacher and is a separate ordinary
capability stratum, not synthetic successful references.

Private per-case artifacts bind sourcecase, original row/trajectory/reference,
donor mapping, each condition content hash/length, canonical boundary state and
aliases, original backend, past observations, original/current source/weights,
actual prompt/input/generated IDs, raw/processed-score definitions, generation
config, fork budget, branch queries/replies, strict scoring and all skips. New
output/run ID, no overwriting; terminal failure retains proof and no automatic
GPU retry. Public Git contains source/spec and reviewed compact aggregates only.

Per-question paired effects are stratified by cell/boundary and compare conditions
on common eligible states, with unmatched/context/terminal skips separately
reported. Deduplicate aliases and keep question as sampling unit (not tokens or
multiple boundaries). Report failures/zero/negative effects and own-rollout versus
same-failed-prefix effects separately. This exploratory64 was selected from the
already-observed192 and does not estimate native512 or seven-benchmark performance.

## 9. Required equivalence/admission fixtures before implementation acceptance

CPU-only scripted model/tokenizer/retriever doubles; no real model/GPU/HTTP needed.
Fixtures must exercise the actual frozen SearchAgent as comparator (lightweight
dependency stubs can load its unchanged definitions) and wrapper using identical
scripted generation/observations. No test mirrors only the wrapper's implementation.

1. Empty state ordinary mode versus native fresh run: exact generated-call context,
   kwargs/stopping start, output response, spans, calls and termination for direct
   answer, one/multiple search and max-turn trajectories.
2. Branch at a search→observation boundary: replay prior observation from artifact;
   no replay HTTP, identical next context, correct t/calls; resumed+replayed result
   equals scripted native complete run. Fork both aliases once when identical.
3. Before-last-policy boundary at t0/t1/t4: forced literal included once only;
   reconstruct forced pending/consumed source gap without deleting a quoted literal.
4. Native HF versus original vLLM forced/context-limit distinction: HF no response
   append on skipped generation; vLLM saved trailing forced gap detected and
   unsupported terminal state skipped, not made into a new generation attempt.
5. Exact threshold: tokens+500=8192 accepted; one over skips, with no generate/post.
   All four prompt conditions may differ in eligibility; common-set counts reconcile.
6. Terminal first answer, closing-answer with malformed think, no-search invalid,
   exhausted-search and loop-budget cannot resume after terminal; a valid earlier
   search state before a future invalid answer remains eligible.
7. Malformed think but native-executable search retains regex behavior; strict EM
   separately rejects final malformed answer. Information/earlier answer tags never
   become final scoring. No semantic/factual label inferred from strict mismatch.
8. Multiple or tag-like observation strings, unclosed generated info, duplicate/
   overlapping/out-of-range spans, missing observation and inconsistent calls/turns:
   exact provenance accepted or explicit rejection/unsupported reason, never guessed.
9. Generated IDs include EOS/overrun text after first closing action: preserve raw
   IDs/pre-cut decode; kept response/span obey native cut; posthoc tokenize differs
   in a deliberately non-round-trip tokenizer fixture and stays separately labeled.
10. Prompt conditions: none uses None, answer deletion only selected final authored
    block, earlier/info answers remain; donor is different qid and mapping stable,
    no available different-answer donor skips rather than claims false reference.
11. Frozen adapter mode/no-reference base-control (CPU scripted distribution): no
    gradient/update requests; source100 zero-B agreement; different ref only changes
    prompt; q native/head causal index and mask tested at chunk31/32/33 boundaries.
12. Native split-encoding versus joint-generation encoding deliberately differ at
    a prefix seam: preserve each definition, no forced identity assertion; processed
    scores change under inherited repetition penalty while raw q is distinguished.
13. Stable files/hash/schema, manifest/donor identities, exclusive output, duplicates,
    partial snapshots and all skips counted. Scripted retrieval failure stops and
    preserves proof; no autonomous retry, signals or launcher action.

Then pilot16 requires actual tokenizer/context-state reconstruction and ordinary-
base comparisons on admitted hardware, with raw/processed score evidence before
completing64. Fixtures alone are not GPU numerical or teacher quality acceptance.
