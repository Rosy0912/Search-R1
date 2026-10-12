# Isolated Ray-IP retry candidate independent review

Result: CODE_CPU_ONLY; no must-fix found in the reviewed service candidate. Reviewer: alignment_generation_review. No runtime source edits were necessary; only a new pure test file and this report were written. No remote operation, deployment, submission, Ray/GPU service start, process/hosts/Slurm action or actual training acceptance occurred.

Exact reviewed service SHA256: 8e38dcbb6a76c62fb14a6d1d3fccbbf47c0496b1182ab0867314e311f95a3ce1

Pure test SHA256: d0ce3c854f2086740b342a2e84c0d29b886eca96e0d21a923d23c842cff03ede

Read-only inspected generator SHA256: 0916b0559535646329d2c5330358ac6cd3e0b248e4b37389c29af9c8ed6b80a0

## Ownership and address review

The isolated helper retains GCS unique local socket owner, actual CLI executable/arguments/epoch, complete sameRUN/job/UID process ancestry and frozen own-service birth checks. The new advertised address must belong to numeric IPv4 addresses returned by actual ip -j -4 address show up; it additionally requires administrativeUP and operationalUP except the standard loopbackUNKNOWN case. Down/inactive addresses, unspecified/multicast/non-numeric values, empty/unknown output are refused. The Ray port remains the separately checked own local listener, not a hostname or arbitrary-IP trust claim.

Connection requires one Alive Ray node and GPU8, valid typed NodeManagerPort, a locally owned raylet executable and CLI ancestry, canonical own short runtime session, second matching raylet port-owner birth observation, repeated current local-address membership and final own GCS/retriever birth validation. Frozen proof retains actual advertised IP/interface evidence, node/port, raylet epoch and session alongside loopback client request address. This fixes the real3348 loopback-string assumption while retaining same-host ownership rather than allowing arbitrary private IPs.

The inspected isolated launch routing uses a new fixed tools/deployment/formal-rayip-v2 prefix and scheduler-head/formal-rayip-v2 receipt namespace while retaining source080, original hardware identity/assets and shared global submission/execution locks. The generator derives candidates from retained3348 originals; reviewed original files are not edited here. This scoped service review is not an exhaustive reacceptance of every generated evidence helper or a target installation result. Root must close all final deployment inventories after any later generated-file change, and actual immutable installation/admission outcomes remain independent records.

## Tests actually executed

Command from tools checkout:

```sh
PYTHONPATH=. ../fresh200-local-test-env/bin/python -B -m pytest -q -p no:cacheprovider reproduction/aligned-em200-deployment/formal-rayip-v2/test_formal_service_proof.py
```

Observed18 passed in0.04s, exit0. Pure fixtures mock the ip command response, Ray client and process ownership observation; they start no service and query no actual target/proc/GPU. Tests exercise the real candidate active-interface filter and verify_ray_connection control flow: normalized local10.33.0.5 acceptance with frozen ownership/session evidence; down/empty/bad IPv4 rejection; nonlocal IP, wrong GPU count, bool/zero/out-of-range port, multiple Alive nodes, wrong executable/CLI ancestry, identity rejection, changed raylet birth, disappearing interface and foreign session rejection. Every rejection refuses proof publication; connected paths always shut down the temporary Ray client.

## Limits

Mocks do not establish target ip utility behavior, actual Linux interface inventory, target socket/proc identity, driver Ray session, GPU leases or real training. Actual3348 is a preserved startup failure, not a successful0-update resume checkpoint or scheduler terminal receipt fabricated after retention loss. A future fresh token/job/RUN must produce its own actual successful ownership proof before unchanged source080 main. The CPU tests are code validation only; hardware3345 acceptance and any later formal first5/200 independent acceptance remain separate facts. No retry was launched by this review.
