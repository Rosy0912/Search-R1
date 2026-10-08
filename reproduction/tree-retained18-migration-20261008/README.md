# Retained18 management route and node migration

This directory records the user-authorized move from retired h100-9 to retained
h100-10 (10.33.0.18). The original source snapshot is e7236b7d8729482aad864419fe986066643f3be5.

The SSH client uses the already present management identity on mgmt-host and
strict existing host-key checks. It never copies key contents or changes global
SSH/VPN/auth configuration. Model, index, dataset and checkpoint bytes must stay
inside the Wuhan H100 cluster; management output is limited to safe metadata.

Current native job3333 is absent from the local wuhan-h100 controller. Its terminal
state and post-parent progress are unproven. A shared copy of original job3331's
rolling4 manifest was found; actual tensor integrity and migration/loading are
still to be established in the authorized migration and real training startup.
This does not prove exact executor resume or preserved post-fourth progress.

Training deployment is not included in this management-only revision. Existing
foreign Slurm allocations, UUID GPU leases and the dual15-percent foreign-resource
limits must be respected. This file is not authorization to cancel other jobs,
submit duplicate training or load checkpoints from other clusters.

The own rolling4 manifest is present on the same-cluster preserved NFS copy with
SHA256 84cf98caf0e7e67b52edf72a3cd57455d58c7e08ef020c4e665d81cf969a3b73.
The migration script copies only required assets and saved-state files to retained18
local storage, verifies saved-state hashes, and creates a task-only compatibility
alias. It does not load tensors, claim exact resume or submit training.
