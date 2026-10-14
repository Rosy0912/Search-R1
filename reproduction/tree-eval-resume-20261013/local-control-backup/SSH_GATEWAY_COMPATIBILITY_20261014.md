# Local management SSH handshake compatibility

This changes only the Mac-to-`mgmt-host` argument list in `ssh_retained18.py`: add `-o KexAlgorithms=curve25519-sha256` and `-o IPQoS=none`. The user's existing `mgmt-host`/en0 configuration, identity selection, timeouts, strict inner host-key checks, `UpdateHostKeys=no`, and retained H100 route stay unchanged. No training/source/service configuration changes.

The current helper and the previous committed backup matched SHA256 `498b50447f37f136ca49cde0cb27ecdcc99db99a425f1b8964de3ac0a231576c`. Exact original bytes are retained in `ssh_retained18.pre_gateway_compat_20261014.py`; the updated backup SHA256 is `ca9599e7349e4ae775ea233246f8aa8718af6ef78a947376f651142750c0cdf7`. Publish this commit and confirm the remote commit before atomically deploying those exact updated backup bytes to the original helper path. Do not replace it in place while another process may import it.

Existing small evidence in `outputs/tree-nested-branch-handoff-20261013/`: phase diagnostic SHA256 `579ab4f00211aa6c06d1f4144420c5a3d75c3d88993837ccdac8b19f94b88495` observed timeout after the server banner and before authentication; one-off compatibility handshake SHA256 `7b6c31ad34373547ff27fbb5634211abe4394e567efecbfc4ec925c6f4d9cb4d` authenticated successfully and returned `hostname=mgmt-host` in3.22s. This is a bounded observed connection result, not a guarantee of future connectivity or evidence of training progress.

Preparation validation: Python AST syntax only. No SSH or training-status read is run by this change. The main controller owns two actual follow-up progress observations after publication/deployment. No credentials or private-key contents are included.
