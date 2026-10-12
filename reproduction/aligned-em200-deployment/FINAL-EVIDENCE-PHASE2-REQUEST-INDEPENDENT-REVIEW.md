# Phase2 installation / Linux fixture request independent review

Scope: CODE_STATIC_ONLY. Reviewer: alignment_generation_review. No installation, remote transport, target execution, fixture rerun, training, GPU, host repair, Slurm or process mutation occurred in this review. No actual final200/official-HF acceptance or independent evidence receipt is issued.

Result: no must-fix found in the frozen request bytes below. This approves the request code for root publication and subsequent observed target execution, not its future result.

## Exact reviewed requests

| Request | Bytes | SHA256 |
| --- | ---: | --- |
| install-final-phase2-v1-part1.py | 38334 | `f66397a3da52b09c0878379cfb017eddddcf71f661e25f9a6c437429616c75cf` |
| install-final-phase2-v1-part2.py | 46564 | `b2d6f812541bbb74884b8d114719aa3364a5d1ecef780d3d50d7fcd0b9b20bd8` |
| install-final-phase2-v1-part3.py | 50912 | `5222ec6c5a5e2e98f943f3c402c8c6d82c816a16ae0406c624c28594bbb75127` |
| install-final-phase2-v1-part4.py | 53762 | `3563c8df7eb0b24a29f4cbe24c78abdbc34d6886ecc08990b6a8908902bc16b5` |
| install-final-phase2-v1-part5.py | 29059 | `a85bf583d793f67bcbfd3cf8a2b8b38be9455c67d35e88781e1072ea383057a0` |
| test-final-phase2-v1-linux.py | 2175 | `f33da3eafda6390a85a4bc8a5c8de931012ce81df01b461f0fbdcab156d7d7ff` |

## Static checks actually performed

Parsed all six requests with Python AST without executing their code. Extracted each embedded deploy.py literal and verified exact equality with the already-reviewed local deploy.py. Decoded all nine base64 payloads with strict validation, verified embedded SHA256, and compared each full byte sequence with Git commit `66d30a6a574a765b3885b12bf0f20703effca08e`. Checked unique payload names, exact nine-file coverage, and equality with the test request's full closure inventory. Largest request is 53762 bytes, below the requested 54000-byte limit.

Installation calls only install_tools; the embedded module name prevents its setup CLI from executing. The reused installer checks Linux h100-5, fixed canonical ROOT, deployment marker, source-isolation record, SOURCE080 clean HEAD and topology. It targets only `ROOT/tools/deployment/final-evidence-phase2-v1` with six dependencies, the tiny test, candidate documentation and prior independent review. Missing files use exclusive xb/fsync; existing matching regular bytes are accepted without rewriting, while differing bytes or redirected paths fail. Existing tools are not overwritten. The inherited installer reads small environment/asset configuration files and filesystem topology/storage metadata; it does not hash model/corpus/index bytes or initialize GPU services. Its prepare() retains the existing >400-GiB free-space requirement, which can prevent installation on a low-space target.

The test request invokes lexical `ROOT/env/bin/python -B -m pytest` with cwd and PYTHONPATH at the new repo, hidden CUDA, offline HF/Transformers, OMP=1, disabled bytecode and pytest cache provider. It names only test_aligned_final_phase2.py; that reviewed file constructs local tiny models and temporary evidence fixtures rather than invoking the phase2 CLI or official asset scan. It checks all nine installed file hashes before and after the child and returns the actual child exit/stdout/stderr under an explicit tiny-fixture-only scope. The 90-second timeout can terminate the child; timeout or missing output must not be interpreted as a passed test.

## Limits

This review does not establish target interpreter/dependency health, Linux fixture results, successful installation, stable resource availability or actual training evidence. Root must preserve the real installation/test responses and require the actual 16-test outcome separately. The five parts are not one atomic transaction: a failed part can leave a partial prefix; the test's full nine-file closure rejects missing or altered members. Filesystem checks assume the existing managed deployment tree has no concurrent hostile writer. Hostname repair remains awaiting separate user authorization and is unrelated to these requests.
