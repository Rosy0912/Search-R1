"""Read the root's already committed fault repair and write a code-only SHA delta.

Never commits, edits the core checkout, imports source, tests or trains.
"""
import argparse,hashlib,json,re,subprocess
from pathlib import Path
BASE='4a65e9a3ab87e84e1e9dd283b664cdef3a224c8a'
HERE=Path(__file__).resolve().parent

def require(value,message):
    if not value:raise ValueError(message)

def git(repo,*args):return subprocess.check_output(['git','-C',str(repo),*args])

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-dir',required=True);p.add_argument('--source-commit',required=True)
    a=p.parse_args();repo=Path(a.source_dir).resolve();commit=a.source_commit
    require(re.fullmatch('[0-9a-f]{40}',commit) and commit!=BASE,'New exact core commit required')
    require(git(repo,'rev-parse','HEAD').decode().strip()==commit,'Root unified HEAD differs')
    require(not git(repo,'status','--porcelain').strip(),'Root source freeze requires the unified committed files')
    names=git(repo,'diff','--name-only',BASE,commit).decode().splitlines()
    require(names,'Actual source fault repair delta required')
    allowed=('search_r1/','verl/')
    require(all(name.startswith(allowed) and Path(name).suffix in ('.py','.yaml','.yml','.json') for name in names),
        'Only code/config core delta may enter V7')
    files={name:hashlib.sha256(git(repo,'show',commit+':'+name)).hexdigest() for name in names}
    result=dict(status='ROOT_UNIFIED_COMMITTED_V7_CODE_ONLY_SOURCE_FROZEN',code_commit=commit,base_code_commit=BASE,
        source_dir=str(repo),changed_source_files_sha256=files,core_source_bytes_changed=True,
        actual_fault='3328 failed third generation: native emitted token IDs/PAD inference assertion',
        continuation_scope='CURRENT_FRESH200_CHAIN_CONTINUATION',parent_job_id=3328,parent_completed_outer=2,
        new_tests_executed=False,new_reviews_executed=False,new_calibration_executed=False,
        contains_model_data_index_checkpoint_assets=False)
    path=HERE/'CORE_FIX_DELTA_V7.json';raw=(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
    with path.open('xb') as stream:stream.write(raw)
    print(json.dumps(dict(delta_path=str(path),delta_sha256=hashlib.sha256(raw).hexdigest(),delta=result),ensure_ascii=False))
if __name__=='__main__':main()
