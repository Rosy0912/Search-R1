# Baseline-aligned retrieval identity gate

The candidate profile must call `validate_retriever_identity(config, tokenizer)` before allocating its training runtime. Existing GPU lease validation remains required. This gate is read-only: it neither acquires GPUs nor starts, restarts, or modifies a running retrieval service.

Configure `retriever.identity_manifest` and `retriever.identity_manifest_sha256` from an independently reviewed frozen baseline receipt. The manifest hash must be supplied separately; deriving it automatically from the same untrusted file would defeat the pin. A receipt prepared from the candidate's current probe responses is not baseline evidence.

The local retrieval service source now offers an opt-in read-only `/identity` endpoint when started with `SEARCH_R1_RETRIEVER_IDENTITY=1`. The candidate change does not restart any existing service. A service without this endpoint fails the aligned gate. The endpoint reads the loaded encoder's actual parameter device/dtype and the FAISS index's actual GPU child device IDs, plus records the FP16/sharded conversion options used by the implementation. Unsupported GPU introspection fails the gate. It records file identities before loading and checks their present identities without rehashing the large index on every request.

The schema is a JSON object with:

- `schema_version`: `1`.
- `assets`: named objects with absolute `path` and pinned `sha256`. Required names are `corpus`, `index`, `service_source`, `identity_source`, and every local encoder file named `encoder/<relative path>`. The source pins refer to the reviewed identity-capable deployment overlay; corpus/index/encoder pins must come from the original baseline assets. Cache and lock files are excluded.
- `topk`: `3`; `max_obs_length`: the frozen observation token budget.
- `format`: the exact JSON string `"Doc N(Title: title) text\\n"`; `observation_truncation`: `"batch_tokenize_then_head"`.
- `runtime`: exact expected fields `retrieval_method`, `topk`, `encoder_path`, `encoder_device`, `encoder_dtype`, `pooling_method`, `query_max_length`, `query_prefix`, `faiss_gpu`, `faiss_gpu_devices`, `faiss_index_class`, `faiss_dimension`, `faiss_ntotal`, `faiss_clone_use_float16`, and `faiss_clone_shard`. E5, `query: `, an actually loaded CUDA encoder, and inspectable GPU FAISS are required. Paths refer to the local service assets. Numeric/dtype/FP16 settings must match the frozen baseline receipt, rather than being inferred from a launch command.
- `probes`: one or more baseline queries, each with `query`, exactly three ordered `contents` strings, exact `formatted_passages`, and `observation_token_ids`. Tokens correspond to the baseline tokenizer after wrapping formatted passages in the archive information block, tokenizing without special tokens, and retaining the first `max_obs_length` tokens.

Preflight hashes each unique asset once by streaming 8 MiB chunks, compares the live service's complete loaded/current file identities with those verified assets, compares actual runtime fields, and submits every pinned probe to the loopback `/retrieve` endpoint. It checks exact ordered contents, formatting, and token truncation. The service's identity response is not itself accepted as a manifest hash or a baseline golden result.

The transport must be plain HTTP on a loopback host and `/retrieve`. The gate preserves existing GPU lease ownership and does not create a new allocation. Index hashing can be slow for the large corpus/index; it is performed once per preflight invocation, not once per query or training step. Changing any asset requires a new independently reviewed manifest and hash.

CPU tests use tiny files and mocked transport to verify manifest/file tampering, loaded/current identity drift, CUDA/FAISS drift, golden order/format/token changes, missing probes, and remote URL rejection. They do not verify real E5 encoding, GPU FAISS ranking, memory use, or throughput. No actual GPU parity or production launch has been performed by these changes; the gate reports `full_gpu_parity_claimed=false` even after its identity and finite probe checks pass. Real baseline-vs-candidate retrieval comparisons remain required on the already authorized hardware.
