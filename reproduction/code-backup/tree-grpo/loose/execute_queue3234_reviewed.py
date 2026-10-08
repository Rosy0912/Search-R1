import pathlib,hashlib,json
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/post3234')
b=(root/'QUEUE_READY_RECEIPT.json').read_bytes()
assert hashlib.sha256(b).hexdigest()=='6c8fb9e8751a4a4886929fca20f92525fef2366e3ba9580088d790bc33ccf26b'
r=json.loads(b);assert r['status']=='READY_FOR_SINGLE_CPU_AFTERANY3234_EVIDENCE_SUBMISSION_ONLY'
p=root/'queue-07b26d81.py';q=p.read_bytes()
assert hashlib.sha256(q).hexdigest()==r['queue_helper_sha256']=='07b26d812db3cf666fd1cb69fde968465d5fe798cd93ed40360bf81c0a19d54f'
exec(compile(q,str(p),'exec'))
