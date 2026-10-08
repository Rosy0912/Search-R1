import hashlib,json,pathlib
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
files=['tools/tree_shared_owned_runtime_fixed.py','tools/resource_guard_identity.py','tools/probe_tree_shared_retrieval.py','tools/check_real_checkpoint.py','repo/reproduction/h20/retriever_local.py']
print(json.dumps({str(root/p):hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files},sort_keys=True))
