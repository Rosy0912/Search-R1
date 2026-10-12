"""B private stage retrieval release. No request to the original oracle."""
import hashlib,json,os,time
from pathlib import Path

def release_retriever_gpu(config,stage):
 settings=config.get('retriever',{});url=settings.get('gpu_release_url')
 if not url:return False
 if url!='http://10.123.4.12:18006/release-gpu':raise ValueError('Unbound B retrieval lifecycle endpoint')
 path=Path(settings['control_token_file'])
 if path.is_symlink() or path.stat().st_uid!=os.getuid():raise ValueError('Unowned retrieval control file')
 binding=Path(settings['phase_binding_file']);raw=binding.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=settings['phase_binding_sha256']:raise ValueError('Phase binding changed')
 b=json.loads(raw)
 if b['host']!='xie-lab-4' or b['accepted'] is not True or b['gpu_shards'] not in (4,8):raise ValueError('Explicit four/eight-GPU phase candidate not admitted')
 if b['gpu_shards']==4 and b.get('cuda_visible_devices')!='0,1,2,3':raise ValueError('Four-GPU phase candidate device boundary changed')
 import requests
 started=time.perf_counter();response=requests.post(url,headers={'x-e1-probe-token':path.read_text().strip()},timeout=(5,180));response.raise_for_status();proof=response.json()
 if proof.get('released') is not True or proof.get('gpu_worker_pid') is not None:raise RuntimeError('Private retrieval worker did not exit')
 previous=proof.get('previous_gpu_worker_pid')
 if previous is not None and Path('/proc',str(previous)).exists():raise RuntimeError('Retrieval worker exit not observed')
 print('RETRIEVER_PHASE='+json.dumps(dict(stage=stage,wall_seconds=time.perf_counter()-started,**proof)),flush=True)
 return True
