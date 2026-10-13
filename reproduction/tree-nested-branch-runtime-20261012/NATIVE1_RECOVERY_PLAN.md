# Same-goal native1 continuation after the 0bef CPU OOM

This candidate resumes only the fresh nested Tree200 goal that started with
`0befbaf99fc44942105831b3fbec94e4bf1b8ccb` on H10018 (`h100-10`). It cannot
load historical base/current3339/current3341 recovery or begin a new fresh run.
The old candidate/source and its evidence remain separate and unchanged.

The old process really completed two outer updates and each rank made 40 Adam
writes. Its CPU cgroup reached 256 GiB and systemd terminated it while native2
was being saved. Native2 has no committed manifest. Therefore the complete
native1 state is the effective recovery boundary: outer1, each rank Adam20,
scheduler1, sampler epoch0/cursor512. The new run recomputes outer2; the original
outer2 and its cost/evidence are retained outside effective optimizer history.
No executor/vLLM equivalence or identical restarted generations are claimed.

`manifest.template.json.nested_native_continuation` contains the protocol and
five path/SHA descriptors: parent native manifest, parent resolved config,
original successful attempt2, failed run terminal, and actual OOM snapshot.
The source pins all five inputs and rejects active, historical, mixed, or
changed identities. `supervisor.py` injects the eleven dedicated
`TREE_NESTED_CHAIN_RESUME_*` fields from this credential. Generic recovery
variables remain forbidden. The runtime argument is the existing config and
run entry; no fresh/Adam0 fallback is permitted.

Before loading state the supervisor compares the whole resolved configuration
with the pinned mother configuration. Only `trainer.default_local_dir` and
`baselines.holdout.output_dir` change to the new child run. The trainer's exact
unused critic schedule injection from -1 to 201 is accounted separately.
Algorithm loss, nested A/A1/B/B1/B2 selection, 512x5 rows, mini128/micro16,
old/ref32, four searches, behavior-mu guard, LR1e-6, warm57, dev5, and milestone
save cadence remain unchanged.

`RayPPOTrainer.fit` observes a cold official bootstrap, then invokes
`tree_nested_chain_resume.migrate_trainer` and the eight-rank
`load_nested_chain_training_state` entry. Existing native worker restoration
loads model shards/parameters/buffers/gradients, Adam, scheduler, and RNG;
existing driver restoration loads sampler order/cursor/generator, driver RNG,
Tree RNG, algorithm stage and complete attempt accounting. Serialized and live
logical hashes must agree before `output/nested-native1-loaded.json` is written.
The first credited child outer is 2. Its fifth-outer live human continuation
and all milestone/evaluation/save behavior are retained.

JSON accounting reads effective outer1 from its original mother paths and
outer2 onward from new child paths, always using the corresponding original
source/config/run identity. It never relabels original evidence as child output
or adds the failed-save outer2 to effective training credit.

Runtime resources admit a 384 GiB own CPU cgroup with at least 448 GiB host
availability; the existing GPU headroom and ownership rules remain unchanged.
The original 0bef algorithm review remains bound. New operational changes do
not claim an independent review; their acceptance comes from the actual native
restore and continuing training measurements. Local validation is restricted
to syntax/JSON parsing and whitespace checking; no GPU test is run locally.
