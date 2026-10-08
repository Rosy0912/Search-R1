"""Source19 metadata compatibility fixture; no actual review or package release."""
import copy,hashlib,importlib.util,json
from pathlib import Path
import tempfile
from unittest.mock import patch
BASE=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
HELP=BASE/'work/tree-fresh200-dispatch-v5'
spec=importlib.util.spec_from_file_location('independent_source19_common',HELP/'fresh200_dispatch_common.py')
C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
maps=json.loads((OUT/'BYTE_MAPS.json').read_bytes())
parent_audit=BASE/'work/tree-fresh200-f7-v8-v5-context-free-review/BYTE_AUDIT.json'
source19=json.loads(parent_audit.read_bytes())['source_files']
assert len(source19)==19 and C.SOURCE_INTERFACE_FILES.issubset(source19)
for name,entry in source19.items():
    raw=(BASE/'work/tree-behavior-mu-ppo'/name).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==entry['sha256'] and entry['matches_git_head'] is True
manifest={'files':{**{'source/repo/'+name:entry['sha256'] for name,entry in source19.items()},
                   **{name:entry['sha256'] for name,entry in maps['wrapper_files'].items()}}}
def sha(raw):return hashlib.sha256(raw).hexdigest()
with tempfile.TemporaryDirectory(dir=OUT,prefix='source19-') as tmp:
    def save(name,value):
        path=Path(tmp)/name;raw=json.dumps(value).encode();path.write_bytes(raw)
        return dict(path=str(path),sha256=sha(raw))
    audit=save('audit.json',dict(git_head=C.COMMIT,source_files=source19,wrapper_files=maps['wrapper_files']))
    context=save('context.json',dict(reviewer_context=dict(fork_turns='none',inherited_parent_conversation=False),
        source=dict(audited_clean_head=C.COMMIT,initial_clean=True),source_interface_reviewed=True,
        stages={'preparation':dict(accepted=True),C.STAGE:dict(source_review_completed=True,source_review_accepted=False)},
        execution_scope=dict(same_h100_9_cpu_only_preparation_allowed=True),artifacts={'BYTE_AUDIT.json':audit['sha256']},
        fixture_only_not_independent_acceptance=True))
    refs=dict(source_interface_receipt={key:maps['source_owner_receipt'][key] for key in ['path','sha256']},
              context_free_review=context,context_byte_audit=audit)
    with patch.object(C,'ROOT',BASE):
        result=C.validate_reviews(refs,manifest)
        assert result['source_review_accepted'] is False
        rejected=[]
        for name in sorted(set(source19)-C.SOURCE_INTERFACE_FILES):
            bad=copy.deepcopy(manifest);bad['files']['source/repo/'+name]='0'*64
            try:C.validate_reviews(refs,bad)
            except ValueError as error:rejected.append(dict(changed_extra_source=name,error=repr(error)))
            else:raise AssertionError('extra audited source not verified: '+name)
    output=dict(status='SOURCE19_REVIEW_METADATA_BOUNDARY_COMPATIBLE',
        actual_source_owner15_bytes_consumed=True,source_owner_receipt=maps['source_owner_receipt'],
        source_audit_file_count=19,wrapper_audit_file_count=15,completed_blocked_fixture_preserves_false=True,
        additional_closure_files_all_verified=True,changed_extra_source_rejections=rejected,
        parent_byte_audit_observed_sha256=sha(parent_audit.read_bytes()),
        fixture_context_not_actual_independent_acceptance=True,actual_gpu_executed=False,actual_remote_executed=False,
        actual_slurm_executed=False,actual_cpu_preparation_executed=False)
    (OUT/'SOURCE19_REVIEW_BOUNDARY_CHECK.json').write_text(json.dumps(output,indent=2,sort_keys=True)+'\n')
