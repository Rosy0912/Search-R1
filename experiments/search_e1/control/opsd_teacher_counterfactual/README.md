# M3 CPU preparation (no evaluation launch)

This is the first implementation layer of the frozen `M3_SPEC.md`. Root reviews,
backs up, publishes and verifies this new source before running real preparation.
It does not change M1, M4 or the305-file runtime, load models/Torch/tokenizer
packages, send HTTP, launch processes, train, or claim teacher benefit.

```sh
python3 -B -m unittest discover -s experiments/search_e1/control/opsd_teacher_counterfactual -p 'test_cpu.py' -v
# Root executes only after publication:
python3 -B prepare_cpu.py --cases-dir /path/to/frozen192-selection \
  --runtime /path/to/frozen305-runtime --output /path/to/NEW-private-preparation \
  --trace-root /path/to/accepted-M1-run
```

`--trace-root` is optional; absent/missing source100 or epoch1 trace artifacts are
explicitly counted, never regenerated. The command never reads full845515 input:
it consumes the already-frozen192-case file (pinned SHA1703b15b…), original sibling/
pair data inside it, and selected existing traces only. File reads are bounded to
128MiB, source hashes/stats and helper hashes/stats are retained/rechecked. The
selected192 contents/semantic hashes/six cells and source core/QA hashes are
verified. Outputs are exclusive/private and contain QA/reference/history text;
they must never be committed. A missing/changed/malformed source or inconsistent
reconstruction fails closed; no partial/unsupported state is silently made ready.
Missing traces and legitimate terminal boundaries are the currently supported
explicit skips. A future extension may represent other unsupported-state skips
after review, rather than weakening preparation checks.

`references.py` takes the first16/cell under the original M1 seed42 hash order,
marking first4/cell as pilot. Exactly64 selected,16 pilot, no all5wrong teachers.
Reference min(search,token_count) keeps original **source line order** ties,
not the case bundle's sibling sorting. Donors are deterministic within-cell cyclic
other-question references with native accepted final answers not matching recipient
gold; no donor means explicit skip. None removes the envelope, not empty text.
Final-answer deletion removes only the accepted final authored complete answer
block. Earlier/information answer mentions remain. Reference hashes are UTF8
content SHA256; object/state/case semantic hashes use canonical JSON. These are
whole-prompt interventions with length/position confounds, not pure answer leaks.

`state.py` has no model/executor invocation. It reconstructs canonical pre-turn
response, original turn/search budgets, forced pending state and past observation
bytes. Its `advance` applies frozen HF post-generation control transitions to
supplied fixture segments/observations; observation must already exist. It does
not fetch/replay it. This transition primitive is ready for a separately reviewed
continuation executor but is not a GPU launcher or model wrapper yet.

Original vLLM states use policy-span gaps and exact known wrappers, preserving
tag-like content inside observations without naive closing-tag truncation. Turn4
forced insertion before vLLM context rejection is distinguished from HF's append
after successful generation. Actual HF traces additionally bind turn context,
policy/cut, executed queries, original reply/top3 and insertion budget, actual IDs
and source event hash. Full actual tokenizer roundtrip and token-count/context
checks remain pending; IDs are never manufactured from kept response text.

Boundary A is after first search **and original observation**, terminal first
actions skip. B is before the last generated policy; un-emitted context-limit
attempts skip instead of relabeling a prior generation. Earlier active states
before a later terminal remain eligible. Same A/B state aliases deduplicate once
per case/history; do not deduplicate unrelated cases using state hash alone.
Empty B histories are marked no failure history; successful CC source histories
are not called failed. No extra five turns or repeated forced text is permitted.

Identical candidate response/spans can match multiple stored siblings. Preparation
retains all matches and uses the native first eligible source-order shadow match;
it does not claim stored-pair origin is uniquely proved. The published M2 selection
audit supplies independent full native pair reconstruction, rather than this
helper repeating full-dataset character distances.

19 small CPU fixtures currently pass without skips. They compile only the actual
unmodified hashed frozen SearchAgent/StopTags class definitions into an isolated
test namespace with scripted tensor/tokenizer/model/request doubles. Thus state/context, structural cut, observations, spans, budgets and termination
are compared against native code, not against a second copy of this helper. Actual
continuation generate kwargs and processor null-effect remain pending. The doubles are
explicit synthetic fixtures, not real generated outputs. Full fixtures cover
private preparation/identity/nooverwrite, selection/donors, source-order ties,
delete-only-final, context+500 threshold, final forced, invalid think/search versus
strict final reward, missing observations, actual HF trace reconstruction, alias
dedup, terminal and fifth-search refusal, EOS/overrun versus posthoc text and
vLLM/HF forced-context divergence.

Real preparation's `accepted_cpu_preparation` means identity-bound CPU files were
prepared, never GPU admission, teacher efficacy or tokenizer equivalence. Root
still checks consumed accepted M1 trace/run identity, actual tokenizer/chat/input
IDs and context budgets for all conditions, source100 zero-B/no-ref control,
frozen teacher q, raw versus processed generation scores, resource/lock/retriever
identity and fixed16 pilot before any actual continuation. Native split-encoded
training q and joint-encoded generation contexts remain distinct as in the spec.
