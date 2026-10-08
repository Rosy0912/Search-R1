# Selected reference and student role audit

This is independent CPU postprocessing of the **accepted** `full-selection-v1`
audit. It uses the existing read-only SQLite spool and its private case/pair
metrics; it never rereads the original rollout or mined-pair inputs, imports a
model, sends HTTP requests, computes new distances, or changes a running job.

For each eligible question, the denominator contains exactly one native
predicted reference sibling and one native full-trajectory-distance predicted
student sibling. `all5_wrong` questions have no correct reference or selected
training pair and contribute zero role rows. The compact report preserves
their excluded-question counts and all six dataset/group populations.

The accepted producer proved its shadow prediction equal to frozen
`core.mine_pairs`. Its exact, raw-input identities remain recorded by the
producer receipt. This helper verifies that the selected sibling occurs among
the stored pair's matching source rows and that its question, answer identity,
response, reward, policy spans and reference costs agree. Stored pairs contain
no sibling ID: multiple matches remain **ambiguous origin evidence**, while the
native predicted role ID remains well-defined. The helper does not claim to
recover a uniquely stored origin.

`strict_em`, final validity and valid-final gold mismatch reuse the accepted
producer's frozen QA/core metrics without changing them. The reference's
original strict correctness is guaranteed by mining, and does not demonstrate
that an actual privileged teacher generation will be correct. Whole transcript
reserved-tag structure is also reused. New nonfinal `<think>` statistics count
a literal stack independently for each original nonfinal authored span; they
are diagnostics, not a replacement for strict final-action parsing. An earlier
unclosed think may coexist with a valid independently authored final answer.

Search-call and termination counts apply only to the selected roles.
`response_char_count` measures characters. `token_count`, when present, is the
stored **full response** token count, with a separate present/missing denominator;
neither length is called a policy-token count. Each role report retains integer
strict-success counts as well as rates.

## Run after source publication

Root reviews, commits, pushes, verifies and deploys these files before running
them. The command requires the independently verified SHA of the accepted
producer summary, never an automatically accepted replacement summary:

```sh
python -B selection_roles.py \
  --selection-run /data/shaoshitong/search-e1-opsd-selection-audit-20261008/runs/full-selection-v1 \
  --expected-summary-sha256 VERIFIED_ACCEPTED_SUMMARY_SHA256 \
  --output /data/shaoshitong/search-e1-opsd-selection-audit-20261008/runs/selected-roles-v1
```

The output directory must be new and receives permission `0700`. SQLite opens
with `mode=ro` plus `PRAGMA query_only=ON`. Pending WAL/journal files, a nonempty
issues artifact, changed source identity, wrong native pair, duplicate IDs,
fabricated hard references, invalid provenance, incomplete coverage or different
cell denominators reject the run; no accepted `summary.json` is written on failure.

`summary.json` contains aggregate counts, definitions, source and artifact
hashes. `roles.private.jsonl` contains only question IDs, selected sibling IDs,
ambiguity and role measures; keep it on the node/private output and **out of Git**.
No original question, answer, response or token text appears in the compact
summary. Every artifact is fresh CPU hashed and its precise stat identity is
checked for changes. The SQLite/pair/case artifact hashes are fresh postprocessing
identities, not a claim that their old contents were previously independently
hashed by the producer. Original raw-line hashes are not replaced by hashes of
canonicalized SQLite JSON.

## CPU fixtures

```sh
python3 -B -m unittest discover -s experiments/search_e1/control/opsd_selection_roles -p 'test_*.py' -v
```

Small SQLite fixtures cover mixed, all-correct and all-wrong groups, exact selected
role denominators, both failure kinds, full-response tokens versus characters,
valid finals with earlier structure defects, ambiguous origin, wrong accepted
summary/path/group/cell/provenance/semantic identities, missing/partial artifacts
and pending spool writes. They do not claim a new real-data scientific result.
