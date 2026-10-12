"""Measure real full-index four-card retrieval before admitting training traffic."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import requests


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--token-file',required=True)
    p.add_argument('--oracle',default='http://10.123.4.11:18003')
    p.add_argument('--candidate',default='http://10.123.4.12:18006');a=p.parse_args()
    base=Path(a.base);token=Path(a.token_file).read_text().strip()
    headers={'x-e1-probe-token':token}
    session=requests.Session();session.trust_env=False
    from search_r1.baselines.recursive_admission_hkust5 import EXPECTED_HEALTH
    oracle=session.get(a.oracle+'/health',timeout=15);oracle.raise_for_status();oh=oracle.json()
    assert all(oh.get(k)==v for k,v in EXPECTED_HEALTH.items())
    response=session.get(a.candidate+'/health',timeout=15);response.raise_for_status();h=response.json()
    assert (h['ntotal'],h['d'],h['gpu_shards'],h['tf32'],h['cpu_parent_cuda_initialized'])==(21015324,768,4,False,False)
    proof=dict(scope='four_gpu_full_scan_and_http_oracle_equality',host='xie-lab-4',gpu_shards=4,
        cuda_visible_devices='0,1,2,3',candidate_pid=h['pid'],candidate_starttick=h['starttick'],
        source_identity_sha256=h['source_identity_sha256'],accepted=False,started_at=time.time(),tests=[])
    queries=['What is the capital of France?']+['History of '+x for x in
        ('London','Paris','Tokyo','Beijing','Rome','Athens','Berlin','Cairo')]*8
    try:
        for selected in (queries[:1],queries):
            body=dict(queries=selected,topk=3,return_scores=True)
            t=time.perf_counter();r=session.post(a.oracle+'/retrieve',json=body,timeout=(10,180));r.raise_for_status();old=r.json();ot=time.perf_counter()-t
            t=time.perf_counter();r=session.post(a.candidate+'/probe/retrieve',json=body,headers=headers,timeout=(10,180));r.raise_for_status();new=r.json();nt=time.perf_counter()-t
            row=dict(queries=len(selected),oracle_seconds=ot,candidate_seconds=nt,
                full_scan=True,complete_json_equal=old==new)
            proof['tests'].append(row);print('FOUR_CARD_RETRIEVAL='+json.dumps(row),flush=True)
            assert old==new, 'Exact score/ID/document HTTP equality failed; candidate remains unadmitted'
            if len(selected)==1:one_response=old
        h=session.get(a.candidate+'/health',timeout=15).json();profile=h['last_scan_profile']
        shapes=profile['actual_worker_shapes']
        assert profile['tiles_merged']==642 and {v['gpu'] for v in shapes}=={0,1,2,3}
        assert sorted(t for v in shapes for t in v['tile_starts'])==list(range(0,21015324,32768))
        gemms=[v for w in shapes for v in w['gemm_shape_counts']]
        assert sum(v['count'] for v in gemms)==1284
        assert all(v['dtype']=='torch.float32' and v['left'][0] in (1,64)
            and v['left'][1]==768 and v['right'][0]==768 and v['right'][1] in (32768,11036) for v in gemms)
        proof['actual_scan_profile']=profile;proof['original_oracle_health']=oh
        proof['first_worker_pid']=h['gpu_worker_pid']
        r=session.post(a.candidate+'/release-gpu',headers=headers,timeout=(10,180));r.raise_for_status()
        proof['release']=r.json();assert proof['release']['released'] and proof['release']['gpu_worker_pid'] is None
        assert not Path('/proc',str(proof['first_worker_pid'])).exists(), 'Joined child must actually exit'
        # A subsequent sampling phase must recreate a different real worker.
        r=session.post(a.candidate+'/probe/retrieve',json=dict(queries=queries[:1],topk=3,return_scores=True),headers=headers,timeout=(10,180));r.raise_for_status()
        assert r.json()==one_response, 'Recreated phase changed the original HTTP result'
        again=session.get(a.candidate+'/health',timeout=15).json()
        assert again['gpu_worker_pid']!=proof['first_worker_pid']
        assert not again['cpu_parent_cuda_initialized']
        proof['second_worker_pid']=again['gpu_worker_pid'];proof['phase_recreated']=True
        proof['accepted']=True;proof['finished_at']=time.time()
    except BaseException as error:
        proof['error']=repr(error);raise
    finally:
        try:
            r=session.post(a.candidate+'/release-gpu',headers=headers,timeout=(10,180));r.raise_for_status()
            proof['final_release']=r.json()
            assert proof['final_release']['released'] and proof['final_release']['gpu_worker_pid'] is None
            if proof.get('second_worker_pid') is not None:
                assert not Path('/proc',str(proof['second_worker_pid'])).exists()
        except BaseException as error:
            proof['accepted']=False;proof['release_error']=repr(error);raise
        finally:
            path=base/'receipts/four-card-full-scan.json';path.write_text(json.dumps(proof,indent=2))
    # Admission follows the real exit check, never precedes an equality/release fault.
    assert proof['final_release']['released'] and proof['final_release']['gpu_worker_pid'] is None
    binding=dict(gate_path=str(path),gate_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (base/'receipts/admitted.json').write_text(json.dumps(binding))
    print('FOUR_CARD_RETRIEVAL_ADMITTED='+json.dumps(dict(gate=str(path),gpu_released=True)),flush=True)


if __name__=='__main__':main()
