# Single-endpoint paired reader: publication and CPU deployment

This is deployment acceptance of the independent CPU reader, not an actual
15/25 endpoint result. No real endpoint request or GPU evaluation was started.

The six files of `control/opsd_signal_weight_endpoint_pair_v1` were each read
from FETCH_HEAD and compared byte for byte with the local files before remote
deployment. Source commit and remote readback are
`64794e8df9775c8074c548222907d44c9f911c99`. Manifest:
`0fef822dfceb89c6b169f2a0f75b6142dc4367f70db21b8504563a080770ab36`.

Actual HKUST4 host was xielab5, UID1001. Immutable helper root:

`/weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261010/source/endpoint-pair-64794e8/opsd_signal_weight_endpoint_pair_v1`

Publication receipt:

`/weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261010/receipts/endpoint-pair-publication-64794e8.json`

Its SHA is `f0fd06e1f9cf4e261d3e8e180778e885f4c48d47233550072e2ff19f75b8bb85`.
All six Node source files matched their published bytes before and after the
actual CPU test process. Direct SSH returned0.

The original-environment Node test CLI returned0: **22 tests, zero skips**.
CUDA_VISIBLE_DEVICES was empty; PYTHONDONTWRITEBYTECODE was1. Synthetic fixtures
exercised the original checkpoint/outer metadata verifier,7d endpoint verifier
and235e paired arithmetic. No production results or numeric optimizer tensors
were substituted. Production trace scoring remains the original strict verifier.

Actual Node test acceptance receipt:

`/weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261010/source/endpoint-pair-64794e8/node22-cpu-acceptance.json`

SHA: `0c5278c3a560252a26d17e48d4a1de7de02e7f728c60e01cce8c513638129c46`.

The three test-only dependency symlinks are in the parent prefix, outside the
helper root. Their targets are the already deployed canonical00fec71 training,
7d9fea2 completion and235e376 analysis helpers. All target source files matched
their exact manifests before test execution. Source-only binding resolves each
test alias to its canonical target. Production requests must use canonical paths,
not these aliases. No original helper/config/active process was changed.

Local retention is limited to code, this document and small publication/test
proofs. The5989bytes of original Node publication and CPU acceptance receipts
were copied byte for byte and independently SHA-checked. No weights, optimizer,
rollout, token, raw evaluation or paired result payload was copied locally.

Future use requires the actual completed raw outer14/24 checkpoint, completeK,
successful native parent/child/worker waits and full512 output. This deployment
does not count toward the eight required evaluations or full100-outer acceptance.
