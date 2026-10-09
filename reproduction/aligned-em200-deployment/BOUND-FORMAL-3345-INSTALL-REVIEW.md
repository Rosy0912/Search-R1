# Hardware3345-bound formal installation review

Result: CODE_STATIC_ONLY; no remaining must-fix found. Reviewer: alignment_generation_review. No builder/request execution, remote installation, submit, service, GPU, hosts or Slurm operation occurred in this review. This review does not independently recertify hardware acceptance or claim formal training has started/completed.

Actual generated bindings:

- External hardware acceptance SHA256: fea2f12a1bbc54008cdc4176d2156105e2988681c435738451dde44200b6d49f
- Existing human-goal launch authorization binding SHA256: 7475bbfb06ec6d1d31589fc24e3ea725466b1a91320b1540692f6e72db83cc24
- Bound admission SHA256: c31f162d85b11de833d05dc70dd0dc71bc888144a2fefe47e3d6b4df4b751bf1
- Preparer SHA256: f55b5ca4c315d97fc1f7b82930f8fbc863e7adacf51b9bbabb2deeba60e4f086

| Actual generated request | Bytes | SHA256 |
| --- | ---: | --- |
| install-bound-formal-part1.py | 13884 | `9ae0a482690abf5d1929c4f5deffffcea32a003c5c72551741fa0457fea844f8` |
| install-bound-formal-part2.py | 20886 | `1bbe40de61f14c5f1e3c176b5a9426826134ac71f76f6f10ae006ced63e0272c` |

Static AST parsing and strict base64 extraction verified each receipt/admission payload equals its generated original SHA. binding.json values equal the full actual bytes. The hardware receipt states the independently accepted hardware3345 scope, fixed SOURCE080/ROOT; this reviewer relies on that separately signed receipt and does not manufacture another hardware PASS. The launch authority records the existing authorized goal, separately from completed training.

The bound admission equals the previously reviewed unbound admission byte-for-byte except exactly three constant substitutions: external hardware SHA, authorization SHA, and exact seven runtime SHA map. All seven runtime files match that map and remain unchanged; admission4bdb and submitd280 originals are independently fixed in the preparer. SOURCE080 remains080ebbc46b89d61567cd33e1a5e2f1b40afd8b84. The admission's full actual hardware/native numeric/live retrieval/resource/terminal/cleanup verification and same-process first5 safeguards remain in place.

Resolved findings: the preparer now requires source080 plus exactly the seven admission runtime names, preventing the older nine-item candidate inventory from installing old admission first or binding an unusable map. The binder source now reads the actual already-installed canonical ROOT/tools/deployment/bind_retrieval_proof.py with fixed SHA3ae4f0095b775716b0f2d3180b513906a7be51837d1f2786b3ec72447733704d; it no longer assumes a nonexistent candidate-prefix copy.

Generated requests check h100-5, canonical ROOT, source080 HEAD and absence of existing formal-submit intent. They install only the two receipt files under ownROOT/logs plus six unchanged candidate runtime files, the same-byte existing binder, unchanged submit helper and newly bound admission under ownROOT/tools/deployment. Existing different bytes are refused, new files use exclusive xb+fsync; modes are explicitly set on matching own files. They contain no submit call or training/service entrypoint. Installation is two-part rather than atomic; partial failure must be retained and inspected. Existing source cleanliness/topology and actual full evidence are rechecked by admission before submission; this code review cannot establish future target state or eliminate concurrent writer races. Actual request results, subsequent bound admission validation and actual submission remain separate observed steps.
