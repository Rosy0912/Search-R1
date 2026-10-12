# Hardware collector independent review

Verdict: **PASS for the reviewed source and CPU fixture scope; no remaining must-fix findings.** This review does not verify target hardware, job3345 completion, retrieval/numerical results, or formal training, and does not constitute a GPU acceptance or release receipt.

Reviewer: `/root/alignment_protocol_data_review`, independent of collector author `/root/alignment_generation_review`. Only `collect_hardware_evidence.py` and `test_collect_hardware_evidence.py` in the deployment-tools checkout were reviewed. The reviewer changed no implementation and performed no remote operation. Uncommitted formal-launch candidates are outside this verdict.

## Final frozen identities

| File | SHA256 |
| --- | --- |
| `reproduction/aligned-em200-deployment/collect_hardware_evidence.py` | `be052023211b4ff49a60a3e3682c7a919ca84140badabc611092b128f2fefab2` |
| `reproduction/aligned-em200-deployment/test_collect_hardware_evidence.py` | `6f19e01683be40f7ae61329ec89d02a3bd0982757ea0c845f3e04f127162d6a5` |

These are the reviewed final bytes after all four findings below were resolved. Publication must preserve these hashes or obtain a follow-up review of changed bytes.

## Verified behavior

- CLI observation is fixed to the original job3345, source `080ebbc46b89d61567cd33e1a5e2f1b40afd8b84`, token `13e07285c97044fd97974d94f53fea1b`, own NFS root/run and h100-5 environment. It checks the target source HEAD. Its external commands read scheduler state; they do not submit, cancel, alter jobs, manage foreign processes, or execute GPU work.
- A successful scontrol observation binds exact job/name/comment, command plus token, working root, requested node and observed node. A purged scontrol record can fall back to the original primary sacct job/name/comment/node. When both exist, identity/state/exit/start/end/submit disagreement refuses terminal classification. Disabled accounting is recorded as an unavailable secondary observation, not inferred as job failure.
- PENDING, RUNNING, COMPLETING and SUSPENDED remain nonterminal. Terminal classification requires a recognized terminal state, actual parseable scheduler EndTime and SubmitTime, ordered end/submit and a numeric exit-code/signal pair; COMPLETED additionally requires an ordered actual StartTime. Unknown/malformed timestamps and foreign identities do not produce a verified terminal archive.
- Terminal output is an exclusive new directory under the own root's `evidence` directory. It retains before/after scheduler observations, original fixed submission records, permitted original run outputs and small deployment/validator code, with copy hashes and an inventory. The original submission receipt's job/source/token/root/comment/run fields are checked. A scheduler transition during copying is retained as an issue.
- Copies reject noncanonical/symlink sources and special files, verify the opened inode against the named original, and compare device/inode/size/mtime/ctime before and after copying. Failed or missing evidence is recorded instead of relabeled as success. Original files remain untouched; an existing archive is never overwritten.
- RUN copying is limited to an explicit artifact allowlist. Unexpected corpus/index/model files are reported and excluded without reading or hashing their contents. Existing golden asset SHA pins are preserved as raw evidence; asset corpora, indices and encoder/model weights are not rehashed by this collector.
- Every observation/archive/result remains `acceptance=False`; the archive explicitly marks `independent_review_receipt=False`. Missing rank files, nonzero job outcomes, copy faults and unavailable records remain visible. An archived terminal failure is factual preservation, not numerical/retrieval/GPU acceptance.

## Findings resolved before signature

1. EndTime previously accepted arbitrary strings and `(null)`. The reviewer reproduced false terminal classification. The final implementation parses actual ISO scheduler timestamps, checks the terminal timeline, and includes rejection fixtures.
2. sacct fallback previously allowed a foreign NodeList with matching job/name/comment. The reviewer reproduced it. Final fallback enforces the allowed original node values and has a foreign-node rejection fixture.
3. Timeout handling previously discarded available partial stdout/stderr. Final handling preserves both raw streams and the observation error, including byte decoding; its counter-test passes.
4. Recursive RUN copying previously could copy/hash an unexpected regular asset. Final explicit artifact selection excludes and reports it before copying; the counter-test asserts the corpus never reaches the copy/hash function.

## Independently executed checks and limits

From the deployment-tools checkout, the reviewer ran:

```text
../fresh200-local-test-env/bin/python -m pytest -q \
  reproduction/aligned-em200-deployment/test_collect_hardware_evidence.py
```

Result: **27 passed in 0.04 seconds**. These tests use simulated scheduler responses and temporary CPU files, including the actual disabled-accounting response shape; they do not contact Slurm or the target host.

The reviewer separately injected a change to an original regular file during `copy_regular`: the actual function rejected it with `evidence changed while copying`, and both the original and failed partial destination remained present. This was a temporary local CPU experiment, not a repository test or production collection.

No real scontrol/sacct query, h100-5 execution, original job3345 evidence archive, numerical fixture, live retrieval check, GPU occupancy check or training run was executed for this review. A later hardware decision must inspect the actual terminal originals and issue a separate independent hardware review; this collector and this source review never issue that receipt.
