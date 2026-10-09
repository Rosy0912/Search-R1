# Formal Ray-IP v2 isolation independent review

Verdict: CODE_CPU_ONLY, no remaining must-fix in the reviewed isolation/admission/evidence-consumer scope at these bytes. No remote operation, submission, deployment, training, GPU inference, large asset hashing or checkpoint reload was performed. The separate service reviewer owns the Ray topology detail review; actual compatibility still requires the new job's evidence.

Reviewed the isolated formal-rayip-v2 candidate against all33 original file-manifest entries from the actual3348 failure archive. Every original SHA was independently rechecked and matches PREVIOUS_FAILURE_FILES. The new admission verifies immutable old files, original startup exit1/cleanup0, own-marker cleanup with no remaining processes, the specific Ray startup error, no attempt/training metrics and no checkpoint directory; it additionally checks absence of the old allocation and old RUN-marked processes before proceeding. A fresh token/job/RUN is exclusively claimed. This is a zero-update new-run retry, not checkpoint resume. SOURCE080 and the original ROOT, model/data/order/golden bindings remain unchanged. Old tools, logs and scheduler receipts are read, never overwritten by this candidate.

Submission records are exclusively written under scheduler-head/formal-rayip-v2 and deployment entries point to tools/deployment/formal-rayip-v2. Both locks retain the shared original ROOT/scheduler-head submission-global.lock and execution-global.lock, so namespace isolation does not allow concurrent own execution. Durable intent precedes sbatch; any existing intent/receipt/response prohibits an unknown-outcome duplicate submission. Worker no-requeue/restart guard, SOURCE080 clean checkout guard, source startup exclusive claim and scoped own-process cleanup remain present.

The initial evidence-consumer finding is resolved: shared aligned_first5_cpu_evidence.identity now requires deployment_directory, submission_namespace and previous_failed_job3348, and rejects3348 itself. The release publisher and both final phases reuse this identity and consume the new scheduler receipt path. Admission.verify_submission independently requires the same namespace fields, actual job/RUN/token, acceptance hashes and exact formal tool pins. The evidence publisher has the new actual deployed helper directory guard. Terminal phase1 requires the new sbatch command path; phase2 binds the new receipt and the original phase1 report. CPU/HF math and audit scopes were not widened into self-acceptance or release.

The generator reads the actual local archive at HERE.parents[2]/evidence/aligned-em200-formal3348-failure-full, checks original hashes and requires a new output directory. The removed dead branch is absent in admission. Current seven FORMAL_FILES pins match the actual entry bytes. Evidence tools retain their full immutable evidence closure and pending work statements; this review covers namespace/path changes, not a repeated mathematical or large-model acceptance.

Independent verification: 29 tests passed (test_retry_namespace.py and test_formal_service_proof.py, local fresh200 Python). Additional read-only shared identity checks accepted the new receipt and rejected all three missing namespace fields plus old3348. All33 actual archived original files were rehashed. No production algorithm source was edited by the reviewer. This report does not authorize launch or certify the zero-update retry as actual successful training.

Reviewed file SHA256 inventory:

```json
{
  "bind_retrieval_proof.py": "3ae4f0095b775716b0f2d3180b513906a7be51837d1f2786b3ec72447733704d",
  "evidence-tools/aligned-first5-math-provenance.json": "a59a0503e2cbb66448e9ddf3b1ef3bc0e7cf7e0d716b68f273f3cfd664435e13",
  "evidence-tools/aligned_final_evidence.py": "26d85e91f4d0b9e636f56fa93193e72edabf117dab8f085b3141a288199f2f41",
  "evidence-tools/aligned_final_phase2.py": "8efbe468a049866f7210b3231afa45f74171ca6ff4ccb801c0eb83f680064213",
  "evidence-tools/aligned_first5_cpu_evidence.py": "4a3666f1671229b869871776918e144e8d2d3ae00edd87224ef682287dce934c",
  "evidence-tools/aligned_first5_math.py": "32c037c9e81bb9b0f3c30396f09623db57a3d763754a5d29b3c28e37964e3815",
  "evidence-tools/collect_hardware_evidence.py": "be052023211b4ff49a60a3e3682c7a919ca84140badabc611092b128f2fefab2",
  "evidence-tools/publish_first5_release.py": "8a7e6c115a0083cff7d467fcc60436ffceae968bce43d9e36692eb6fd8b037f1",
  "formal.sbatch": "234a5a625d795151c6718698ce9b6803f39685584c2ed6a55300b215d3ae7f39",
  "formal_admission.py": "0f6053856e619b8d8b2b8a9b29664923ffc093522f635ba154623526e1f0f8be",
  "formal_driver.py": "6c32119ec6e0267c869f9fb035ce584f6cc867bab88583b4e20fd3e2e14c943a",
  "formal_inner.sh": "4c2c65cc4a8c54da78f3eaef90f2dd2f11bd9f584b48a9d786c427d0b98a426b",
  "formal_service_proof.py": "8e38dcbb6a76c62fb14a6d1d3fccbbf47c0496b1182ab0867314e311f95a3ce1",
  "formal_worker.sh": "2e4b9a20c4accf647179706e282ffc1a26334e0db1dd4fdb4bd1b93ac4c0b35e",
  "isolation-manifest.json": "42cff95333cac3d04c4d9e1f2955fda5f3a22d09fe67661eab9863f76171ea20",
  "ray_cli_compat.py": "a6ea15b2ac65ff347ca574ed3af9688c22777723d6a2ddbd7b88f33455f6ddc3",
  "submit_formal.py": "9ccdb868b6bce69d04adff621ff695585d9bc327ad4ad5dd2564fecf3c8e64be",
  "test_formal_service_proof.py": "d0ce3c854f2086740b342a2e84c0d29b886eca96e0d21a923d23c842cff03ede",
  "test_retry_namespace.py": "01e494e6ddd41ff6869fc55bbec9fa5dffd75745826544da07aaa475f6be68a0"
}
```
