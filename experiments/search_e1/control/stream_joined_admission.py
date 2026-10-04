import subprocess,json,time,os,urllib.request
from pathlib import Path
b=Path(os.environ["E1_BASE"]);old=Path("/weka_3/shaoshitong/search-e1-config-study-20261004")
r=json.loads((old/"receipts/archive-round-1-grpo.json").read_text())
assert r["run_id"]=="f04fe593830c4831918ede872812c016" and r["status"]=="RUNNING" and time.time()-r["updated_unix"]<90
worker=Path("/proc/1438383");service=Path("/proc/1705547")
assert worker.stat().st_uid==service.stat().st_uid==os.getuid()
assert b"E1_RUN_ID=f04fe593830c4831918ede872812c016" in worker.joinpath("environ").read_bytes().split(b"\0")
original="/weka_3/shaoshitong/search-e1-retriever-stream-20261004"
assert ("E1_BASE="+original).encode() in service.joinpath("environ").read_bytes().split(b"\0")
assert (original+"/source/retriever_stream.py").encode() in service.joinpath("cmdline").read_bytes()
health=json.load(urllib.request.urlopen("http://10.123.4.11:18002/health",timeout=10))
assert health["acceptance_sha256"]=="75c4a3b2e60bdfb0d1c450e01f9cd99bec2d5245d753d28704755b1d08fdb30d"
uuid="GPU-21352a4a-5b55-8adf-8e82-c70b7a94ecf3"
apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid,used_memory","--format=csv,noheader,nounits"],text=True)
selected=[x.split(",") for x in apps.splitlines() if x.startswith(uuid)]
assert {int(x[1]) for x in selected}=={1438383,1705547},selected
assert all(int(x[2])<=1024 for x in selected if int(x[1])==1705547)
assert sum(int(x[2]) for x in selected)<49140-7168,selected
assert 39574+610+1024<49140-6144
(b/"receipts/owned-shared-gpu-admission.json").write_text(json.dumps(dict(run=r["run_id"],worker=1438383,old_service=1705547,uuid=uuid,apps=apps,time=time.time()),indent=2))
print("OWNED_GPU7_SHARED_ADMISSION_PASSED",flush=True)
