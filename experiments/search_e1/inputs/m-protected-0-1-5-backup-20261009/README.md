# M initial / update 1 / update 5 copy

This fixed operational entry reuses the published C1 protected-state inventory,
source-metadata and full local-SHA verification scripts. Only the actual M
paths, immutable pins, run/config identity and backup schema are changed.

Root's native read returned actual rc0 for all three manifests and the M config;
its preserved proof is `goal-continuation-20261009-21/M-protected-actual-pins/actual-pins.private.json`.
The pinned states are update0, update1 and update5, each at outer0. Each manifest
declares itself plus one shared optimizer, eight rank RNG/cursor/scaler files and
three adapter files (13 members per state, 39 in total). These counts and declared
sizes are derived from the actual manifests; no payload SHA readback, transfer,
numeric Adam audit or backup completion is claimed by this preparation.

Run the published inventory bytes through the existing direct SSH route
(`/usr/bin/ssh -B en0 -oProxyCommand=none -oProxyJump=none -oControlMaster=no -S none`),
derive the fixed native rsync files-from list and copy to a new independent
`m-protected-0-1-5-closed-backup-20261009` destination with no deletion. Preserve
actual transfer rc/stdout/stderr. After real rc0, recheck source file-set and
metadata, then independently fresh-read all local member SHA using `verify-local.py`
before recording completion. The original published `backup.py` SHA is bound in
all three scripts; the Node inventory itself uses fresh file SHA and stat stability.

The existing outer-end compact queue remains independent. This entry neither
retransfers C0/C1 protected states nor claims their acceptance for M. It does not
load models, use GPU/HTTP, write Node files, signal processes, delete original
assets or take training locks. Backup remains outside the training dependency.
Numeric optimizer acceptance is a separate proof. The completed first M outer
and eleven actual updates are not substituted for that proof.

READY means only local source preparation and three static AST checks. Root must
publish/read back these exact bytes before running any entry. Actual transfer,
source metadata-after and all local member SHA remain PENDING.
