"""Pure local exact-byte/AST binding audit; does not import runtime/source code."""
import ast
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
V4=BASE/'tree-fresh200-train-runtime-v4'
V3=BASE/'tree-fresh200-train-runtime-v3'
SOURCE=BASE/'tree-behavior-mu-utf8-v2'
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
pins={
 'v4_inventory':(V4/'FILE_HASHES_V4.json','ba118bdd4af566205c7c776ce8d76e8d6d5bd16e7c15cb135ad086cfd7be4408'),
 'v4_owner':(V4/'SOURCE_CPU_RECEIPT_V4.json','2e9372462bf4bbc8024a381a592ac0a894d82a867a94183f6dfdb10c760dfec6'),
 'source_adapter':(V4/'SOURCE_INTERFACE_COMPATIBLE_RECEIPT.json','455fc4519b4baa36b8faf73946f993c87a5f34ed2c8d36f608b24c2c03d3a8bb'),
 'v4_patch':(V4/'V4_CHANGES.patch','ec7f8abf6e11927d8656395399574428d702365fff3e0e038930ef3e4692dc7c'),
 'source_owner':(BASE/'tree-fresh200-utf8-v2-owner/UTF8_V2_SOURCE_INTERFACE_CPU_RECEIPT.json','5648c41dc787f95d750157fd876194182c6812665b6e4039db31a0ef86140ce2'),
 'source_independent_cpu':(BASE/'tree-fresh200-02b1-v10-v10-context-free-review/RECEIPT.json','c29fdcb835263b9cf03705cabe5cca80903aba72c20c07e797b8cc70e4e621c5'),
 'v3_rejection_receipt':(BASE/'tree-fresh200-formal-v3-02b1-context-free-review/RECEIPT.json','b7590e08e7a2874f8ab5a086c2bc4cc739d62731c73f660abe0b6eb2e1ca499d'),
 'v3_rejection_review':(BASE/'tree-fresh200-formal-v3-02b1-context-free-review/REVIEW.json','a86a4203fda99adbbed6a13e7355af560acf32936c080a34833b95918b4ddcdb')}
bindings={name:dict(path=str(path),expected_sha256=expected,actual_sha256=digest(path),matches=digest(path)==expected)
          for name,(path,expected) in pins.items()}
assert all(x['matches'] for x in bindings.values())
inventory=json.loads(pins['v4_inventory'][0].read_text())
owner=json.loads(pins['source_owner'][0].read_text())
adapter=json.loads(pins['source_adapter'][0].read_text())
wrappers={name:dict(path=str(V4/name),sha256=digest(V4/name),matches_inventory=digest(V4/name)==expected,
                    unchanged_from_v3=(V3/name).read_bytes()==(V4/name).read_bytes())
          for name,expected in inventory['wrapper_files'].items()}
all_files={name:dict(sha256=digest(V4/name),matches_inventory=digest(V4/name)==expected)
           for name,expected in inventory['files'].items()}
assert len(wrappers)==33 and all(x['matches_inventory'] for x in wrappers.values())
assert all(x['matches_inventory'] for x in all_files.values())
changed=sorted(name for name,value in wrappers.items() if not value['unchanged_from_v3'])
assert changed==sorted(('check_behavior_mu_cpu.py','launch.sbatch','sitecustomize.py','supervisor.py','slurm_identity.py','guard_entry.py','submit_once.py'))
assert all(adapter[key]==value for key,value in owner.items())
assert set(adapter)-set(owner)=={'actor_continue_protocol','additional_changed_runtime_files','required_new_interface_key','metadata_adapter'}
assert adapter['additional_changed_runtime_files']==owner['additional_runtime_files']
source_closure={**owner['source_files'],**owner['additional_runtime_files']}
assert len(owner['source_files'])==15 and len(source_closure)==19
sources={name:dict(path=str(SOURCE/name),sha256=digest(SOURCE/name),matches_owner=digest(SOURCE/name)==expected,
                    unchanged_from_original_f7=digest(SOURCE/name)==owner['original_f7_source_files_byte_sha256'][name])
         for name,expected in source_closure.items()}
assert all(x['matches_owner'] for x in sources.values())
source_delta=sorted(name for name,value in sources.items() if not value['unchanged_from_original_f7'])
assert source_delta==['search_r1/baselines/tree_raw_tokens.py','verl/workers/fsdp_workers.py']
protocol=ast.parse((SOURCE/'search_r1/baselines/tree_guard_continue.py').read_text())
assert any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PROTOCOL' for t in n.targets)
           and isinstance(n.value,ast.Constant) and n.value.value==adapter['actor_continue_protocol'] for n in protocol.body)
runtime_entries={name:dict(path=value['path'],sha256=value['sha256'],matches_v4_wrapper=value['sha256']==inventory['wrapper_files'][name])
                 for name,value in inventory['runtime_entries'].items()}
result=dict(all_pinned_input_sha_verified=True,byte_closure_verified=True,input_bindings=bindings,
            wrapper_count=33,wrapper_records=wrappers,full_inventory_count=len(all_files),full_inventory_records=all_files,
            changed_wrapper_count=len(changed),changed_wrapper_files=changed,unchanged_wrapper_count=33-len(changed),
            source_closure_count=19,source_interface_count=15,source_records=sources,source_delta=source_delta,
            source_math_guard_accounting_files_unchanged=True,
            source_adapter_original_fields_exactly_preserved=True,source_adapter_added_aliases_only=True,
            source_adapter_protocol_checked_via_actual_ast=True,inventory_runtime_entries=runtime_entries,
            stale_runtime_entry_sitecustomize=True,
            stale_runtime_entry_effect='Informational inventory inherited from V3; builder requires supplied source.runtime_entries to match wrapper_files and rejects old V3 bytes. Correct future input descriptors; no actual package was built.',
            reviewer_operations='Pure file reads/SHA/AST only; no source import, subprocess, asset, PT, network, scheduler, GPU or formal execution.')
(HERE/'EXACT_BYTE_AUDIT_RESULT.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(pinned_inputs=len(bindings),wrapper_count=33,full_inventory_count=len(all_files),
                      source_closure_count=19,changed_wrapper_count=len(changed),all_hashes_match=True)))
