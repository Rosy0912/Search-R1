# HPC3 environment and training preparation

This package prepares an isolated HPC3 runtime; it does not start GPU training.
FA3 source: v2.7.4.post1, commit5231d95fe13733fb534c01895f7ea88c6a6c7793; CUTLASS c506e16788cb08416a4a57e11a9067beeee29420.
Dependencies use the HPC3 internal repository. Packages, wheels and all large artifacts remain on remote shared storage.
FA3 native import passed with the CUDA12.8 runtime preload. The compatibility package build and HPC3 runtime port are still undergoing validation; this snapshot does not claim training readiness.
The process controller uses a child subreaper and stopped-parent ancestry checks on the HPC3 kernel without pidfd. Its actual CPU Slurm integration test verifies orphan cleanup, TERM escalation and protection of baseline processes.

The HPC3 cleanup backend serializes controller Python child creation and reaping while stopping owned parent trees before identity-checked signalling. Actual CPU Slurm tests cover guardian-thread cleanup concurrent with Popen polling, protected baseline descendants, orphan adoption, TERM resistance, and resuming stopped owned children after refusing an unproven ancestry. Training is not started by preparation.
