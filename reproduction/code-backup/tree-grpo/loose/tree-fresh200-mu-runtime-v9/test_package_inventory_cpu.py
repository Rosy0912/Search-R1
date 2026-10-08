"""Real fifteen-wrapper-byte packaging plus synthetic review metadata boundaries.

Only local code files are used. The tiny Git source fixture contains copied
actual API source and fake asset-shaped files; it is not full training source,
independent review, model/RPC execution or dispatch permission.
"""
import ast
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import build_package_draft as B

HERE=Path(__file__).resolve().parent
COMMON=HERE.parent/'tree-fresh200-dispatch-v5/fresh200_dispatch_common.py'
spec=importlib.util.spec_from_file_location('actual_fresh200_review_boundary',COMMON)
C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PackageInventoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='.package-cpu-',dir=HERE)
        self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.inventory=dict(files={name:dict(sha256=sha(HERE/name),bytes=(HERE/name).stat().st_size)
                                   for name in B.WRAPPER_FILES})
        self.inventory['wrapper_files']={name:record['sha256'] for name,record in self.inventory['files'].items()}
        self.inputs=json.loads((HERE/'INPUTS.schema.json').read_bytes())
        self.repo=self.root/'tiny-source';self.repo.mkdir()
        actual=HERE.parent/'tree-behavior-mu-ppo'
        source_names=sorted(C.SOURCE_INTERFACE_FILES)
        for name in source_names:
            target=self.repo/name;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes((actual/name).read_bytes())
        self.excluded={'example/corpus.jsonl':b'{"synthetic_cpu":true}\n',
                       'example/wiki-18.jsonl.gz':b'synthetic not gzip',
                       'example/e5_Flat.index':b'synthetic index',
                       'public/test.png':b'synthetic image',
                       'reproduction/h20/outputs/old-a800-audit.json':b'{"synthetic_old_cluster_audit":true}',
                       'reproduction/baselines/periodic/old-trajectories.json':b'{"synthetic_old_trajectory":true}',
                       'runs/old.json':b'{"synthetic_run":true}',
                       'tests/fixture.json':b'{"synthetic_test":true}',
                       'model.json':b'{"synthetic_root_model":true}'}
        for name,raw in self.excluded.items():
            target=self.repo/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        for name in B.SOURCE_ROOT_FILES:
            (self.repo/name).write_text('# synthetic allowed runtime packaging root\n')
        subprocess.run(['git','init','-q'],cwd=self.repo,check=True)
        subprocess.run(['git','add','.'],cwd=self.repo,check=True)
        subprocess.run(['git','-c','user.name=Local CPU fixture','-c','user.email=cpu-fixture@example.invalid',
                        'commit','-qm','Copied source-only fixture'],cwd=self.repo,check=True)
        self.commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=self.repo,text=True).strip()
        self.inputs['source'].update(path=str(self.repo),code_commit=self.commit,
            source_interface_receipt_sha256='a'*64)
        self.input_path=self.root/'inputs.json';self.input_path.write_text(json.dumps(self.inputs))
        self.inv_path=self.root/'inventory.json';self.inv_path.write_text(json.dumps(self.inventory))
        self.package=self.root/'package'
        command=[sys.executable,str(HERE/'build_package_draft.py'),'--inputs',str(self.input_path),
                 '--output',str(self.package),'--wrapper-inventory',str(self.inv_path),
                 '--wrapper-inventory-sha',sha(self.inv_path)]
        completed=subprocess.run(command,text=True,capture_output=True)
        self.assertEqual(completed.returncode,0,completed.stderr)
        self.manifest=json.loads((self.package/'manifest.json').read_bytes())

    def review_inputs(self):
        def save(name,value):
            path=self.root/name;path.write_text(json.dumps(value));return dict(path=str(path),sha256=sha(path))
        source={name[len('source/repo/'):]:digest for name,digest in self.manifest['files'].items()
                if name.startswith('source/repo/')}
        interface=dict(status='FRESH200_REPAIRED_SOURCE_INTERFACE_CPU_OWNER_VERIFIED',
                       code_commit=self.commit,source_clean=True,actual_gpu_executed=False,
                       actual_remote_executed=False,training_release=False,source_files=source,fixture_only_not_independent_review=True)
        audit=dict(git_head=self.commit,source_files={name:dict(sha256=digest,matches_git_head=True)
                   for name,digest in source.items()},wrapper_files={name:dict(sha256=sha(HERE/name),matches=True)
                   for name in B.WRAPPER_FILES},fixture_only_not_independent_review=True)
        audit_entry=save('audit.json',audit)
        context=dict(reviewer_context=dict(fork_turns='none',inherited_parent_conversation=False),
                     source=dict(audited_clean_head=self.commit,initial_clean=True),source_interface_reviewed=True,
                     stages={'preparation':dict(accepted=True),C.STAGE:dict(source_review_completed=True,source_review_accepted=False)},
                     execution_scope=dict(same_h100_9_cpu_only_preparation_allowed=True),
                     artifacts={'BYTE_AUDIT.json':audit_entry['sha256']},fixture_only_not_independent_review=True)
        return dict(source_interface_receipt=save('interface.json',interface),
                    context_free_review=save('context.json',context),context_byte_audit=audit_entry)

    def validate(self,inputs,manifest,dispatch=False):
        with patch.object(C,'ROOT',self.root),patch.object(C,'COMMIT',self.commit):
            return C.validate_reviews(inputs,manifest,for_dispatch=dispatch)

    def test_actual_builder_copies_all15_bytes_and_complete_review_boundary(self):
        self.assertEqual(set(self.manifest['wrapper_files']),set(B.WRAPPER_FILES))
        self.assertEqual(len(self.manifest['wrapper_files']),15)
        for name in B.WRAPPER_FILES:
            self.assertEqual((self.package/name).read_bytes(),(HERE/name).read_bytes())
            self.assertEqual(self.manifest['files'][name],sha(HERE/name))
        inputs=self.review_inputs()
        self.assertIs(self.validate(inputs,self.manifest)['source_review_accepted'],False)
        with self.assertRaisesRegex(ValueError,'remains blocked'):self.validate(inputs,self.manifest,True)

    def test_original_nine_file_inventory_reproduces_h01_rejection(self):
        missing=('PLAN.md','SOURCE_CPU_RECEIPT.json','build_package_draft.py','render_dispatch.py',
                 'test_runtime_contract_cpu.py','tokenizer_raw_preflight.py')
        original=copy.deepcopy(self.manifest)
        for name in missing:del original['files'][name]
        self.assertEqual(len(set(B.WRAPPER_FILES)&set(original['files'])),9)
        with self.assertRaises((KeyError,ValueError)) as raised:self.validate(self.review_inputs(),original)
        evidence=dict(status='ORIGINAL_NINE_FILE_PACKAGE_REJECTED_BY_ACTUAL_VALIDATE_REVIEWS',
            complete_wrapper_inventory_count=15,old_wrapper_inventory_count=9,missing_files=list(missing),
            actual_helper_sha256=sha(COMMON),exception=repr(raised.exception),fixture_review_not_dispatch_permission=True)
        path=HERE/'H01_REPRODUCED_CPU.json'
        if not path.exists():path.write_text(json.dumps(evidence,indent=2)+'\n')

    def test_each_reviewed_wrapper_hash_or_missing_item_rejects(self):
        inputs=self.review_inputs()
        for name in B.WRAPPER_FILES:
            bad=copy.deepcopy(self.manifest);bad['files'][name]='0'*64
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,'runtime differs'):
                self.validate(inputs,bad)
        bad=copy.deepcopy(self.inventory);del bad['wrapper_files']['PLAN.md']
        with self.assertRaisesRegex(ValueError,'all15'):B.prepare_wrapper_inventory(bad,self.inputs)

    def test_four_entries_require_actual_inventory_sha_regular_file_and_same_bytes(self):
        for name in B.RUNTIME_ENTRIES:
            supplied=copy.deepcopy(self.inputs);supplied['runtime_entries'][name]['sha256']='0'*64
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,'entry SHA'):
                B.prepare_wrapper_inventory(self.inventory,supplied)
        bad=copy.deepcopy(self.inputs);del bad['runtime_entries']['sitecustomize.py']
        with self.assertRaisesRegex(ValueError,'Missing fresh'):B.prepare_wrapper_inventory(self.inventory,bad)
        link=self.root/'entry-link.sh';link.symlink_to(HERE/'run_tree.sh')
        bad=copy.deepcopy(self.inputs);bad['runtime_entries']['run_tree.sh']['path']=str(link)
        with self.assertRaisesRegex(ValueError,'regular file'):B.prepare_wrapper_inventory(self.inventory,bad)
        wrong=self.root/'wrong.sh';wrong.write_text('different local code\n')
        bad=copy.deepcopy(self.inputs);bad['runtime_entries']['run_tree.sh']['path']=str(wrong)
        with self.assertRaisesRegex(ValueError,'bytes differ'):B.prepare_wrapper_inventory(self.inventory,bad)

    def test_final_inventory_record_schema_sha_and_size_are_required(self):
        prepared=B.prepare_wrapper_inventory(self.inventory,self.inputs)
        self.assertEqual(set(prepared),set(B.WRAPPER_FILES))
        bad=copy.deepcopy(self.inventory)
        bad['files']={name:record['sha256'] for name,record in bad['files'].items()}
        with self.assertRaisesRegex(ValueError,'SHA/byte record'):B.prepare_wrapper_inventory(bad,self.inputs)
        for field,value in [('sha256','0'*64),('bytes',True),('bytes',-1),('bytes','100')]:
            bad=copy.deepcopy(self.inventory);bad['files']['supervisor.py'][field]=value
            with self.subTest(field=field,value=value),self.assertRaisesRegex(ValueError,'SHA/byte record'):
                B.prepare_wrapper_inventory(bad,self.inputs)
        bad=copy.deepcopy(self.inventory);bad['files']['supervisor.py']['bytes']+=1
        with self.assertRaisesRegex(ValueError,'byte count'):B.prepare_wrapper_inventory(bad,self.inputs)

    def test_positive_code_selection_excludes_asset_shapes_and_records_hashes(self):
        selection=self.manifest['source_selection']
        self.assertEqual(selection['excluded_count'],len(self.excluded))
        self.assertEqual(selection['protocol'],'runtime_review_code_metadata_allowlist_v2')
        self.assertEqual(selection['allowed_source_prefixes'],['search_r1/','verl/'])
        self.assertEqual(selection['allowed_source_root_files'],sorted(B.SOURCE_ROOT_FILES))
        self.assertTrue(set(C.SOURCE_INTERFACE_FILES).issubset(selection['selected_files']))
        self.assertEqual(selection['excluded_files'],{name:hashlib.sha256(raw).hexdigest()
                                                      for name,raw in self.excluded.items()})
        for name in self.excluded:self.assertFalse((self.package/'source/repo'/name).exists())
        for name in ('weights.bin','model.safetensors','data.parquet','vectors.npy','wiki-18.jsonl',
                     'wiki-18.jsonl.gz','e5_Flat.index','table.csv','public/chart.png',
                     'reproduction/a800/audit.json','reproduction/run/periodic/trajectories.json',
                     'tests/fixtures/trace.json','model.json','receipt.json','note.py','docs/note.md'):
            self.assertFalse(B.source_file_allowed(name),name)
        for name in (*sorted(B.SOURCE_ROOT_FILES),'search_r1/note.py','search_r1/config.yaml',
                     'verl/metadata.json','verl/launch.sbatch','verl/version/version','verl/single_controller/version/version'):
            self.assertTrue(B.source_file_allowed(name),name)

    def test_source_archive_links_traversal_duplicates_and_special_files_refuse(self):
        link=tarfile.TarInfo('example/e5_Flat.index');link.type=tarfile.SYMTYPE;link.linkname='/outside'
        traversal=tarfile.TarInfo('../outside.py');traversal.size=1
        special=tarfile.TarInfo('special.py');special.type=tarfile.FIFOTYPE
        duplicate=tarfile.TarInfo('duplicate.py');duplicate.size=1
        cases=([(link,None)],[(traversal,b'x')],[(special,None)],[(duplicate,b'x'),(duplicate,b'y')])
        for index,members in enumerate(cases):
            raw=io.BytesIO()
            with tarfile.open(fileobj=raw,mode='w') as archive:
                for member,payload in members:archive.addfile(member,None if payload is None else io.BytesIO(payload))
            output=self.root/('unsafe-%d'%index);output.mkdir()
            with self.subTest(case=index),self.assertRaises(ValueError):B.materialize_source_archive(raw.getvalue(),output)

    def test_actual_frozen_tracked_runtime_closure_dry_selection(self):
        """Read Git code/text metadata only, never build an actual-source package."""
        repo=HERE.parent/'tree-behavior-mu-ppo'
        commit=C.COMMIT
        self.assertEqual(subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),commit)
        self.assertEqual(subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True),'')
        listing=subprocess.check_output(['git','ls-tree','-r','--long','-z',commit],cwd=repo)
        selected,excluded={},{}
        text_records=[]
        for item in listing.split(b'\0'):
            if not item:continue
            metadata,name=item.decode().split('\t',1)
            mode,kind,oid,size=metadata.split()
            self.assertEqual(kind,'blob');self.assertIn(mode,('100644','100755'))
            record=dict(git_blob_oid=oid,bytes=int(size))
            if B.source_file_allowed(name):
                self.assertEqual(str(C.relative_file('source/repo/'+name)),'source/repo/'+name)
                selected[name]=record;text_records.append((name,record,True))
            else:
                with self.assertRaises(ValueError):C.relative_file('source/repo/'+name)
                excluded[name]=record
                # Audit/code text hashes are local evidence. Binary, JSONL,
                # compressed data and tensor bodies are never read or copied.
                if Path(name).suffix.lower() in B.CODE_SUFFIXES or name in B.CODE_EXTENSIONLESS:
                    text_records.append((name,record,False))
        batch=subprocess.run(['git','cat-file','--batch'],cwd=repo,
            input=''.join(record['git_blob_oid']+'\n' for _,record,_ in text_records).encode(),
            capture_output=True,check=True).stdout
        cursor=0;read_bytes=0;external_local_imports=[]
        top_paths={name.split('/')[0] for name in selected}|{name.split('/')[0] for name in excluded}
        for name,record,is_selected in text_records:
            end=batch.index(b'\n',cursor);header=batch[cursor:end].decode().split()
            self.assertEqual(header,[record['git_blob_oid'],'blob',str(record['bytes'])])
            cursor=end+1;raw=batch[cursor:cursor+record['bytes']];cursor+=record['bytes']
            self.assertEqual(batch[cursor:cursor+1],b'\n');cursor+=1;read_bytes+=len(raw)
            record['sha256']=hashlib.sha256(raw).hexdigest()
            if is_selected and name.endswith('.py'):
                for node in ast.walk(ast.parse(raw,filename=name)):
                    modules=([alias.name for alias in node.names] if isinstance(node,ast.Import) else
                             [node.module] if isinstance(node,ast.ImportFrom) and node.level==0 and node.module else [])
                    for module in modules:
                        first=module.split('.')[0]
                        if first in top_paths and first not in ('search_r1','verl'):
                            external_local_imports.append(dict(source=name,module=module))
        self.assertEqual(cursor,len(batch))
        self.assertFalse(external_local_imports,'Selected source statically imports an excluded local package')
        self.assertTrue(C.SOURCE_INTERFACE_FILES.issubset(selected))
        for name in ('search_r1/baselines/preflight.py','verl/trainer/main_ppo.py',
                     'verl/third_party/vllm/vllm_v_0_6_3/llm.py',
                     'verl/version/version','verl/single_controller/version/version'):
            self.assertIn(name,selected)
        self.assertTrue(any(name.startswith('reproduction/') and name.endswith('.json') for name in excluded))
        self.assertTrue(any('a800' in name and name.endswith('.json') for name in excluded))
        inputs=json.loads((HERE/'INPUTS.schema.json').read_bytes())
        self.assertEqual(inputs['external_files'],{str(C.ROOT/name):value for name,value in C.EXTERNAL_RELATIVE_FILES.items()})
        local_external={
            'repo/reproduction/h20/retriever_local.py':repo/'reproduction/h20/retriever_local.py',
            'tools/check_real_checkpoint.py':HERE.parent/'check_real_checkpoint.py',
            'tools/probe_tree_shared_retrieval.py':HERE.parent/'probe_tree_shared_retrieval.py',
            'tools/resource_guard_identity.py':HERE.parent/'resource_guard_identity.py',
            'tools/tree_shared_owned_runtime_fixed.py':HERE.parent/'tree_shared_owned_runtime_fixed.py'}
        local_hashes={name:sha(path) for name,path in local_external.items()}
        self.assertEqual(local_hashes,C.EXTERNAL_RELATIVE_FILES)
        report=dict(status='LOCAL_FROZEN_RUNTIME_PATH_DRY_CODE_SELECTION_OBSERVED_NOT_RELEASED',
            code_commit=commit,tracked_count=len(selected)+len(excluded),selected_count=len(selected),
            excluded_count=len(excluded),selected_files=selected,excluded_files=excluded,
            local_code_or_audit_text_bytes_hashed=read_bytes,
            excluded_binary_asset_bodies_read=False,actual_model_asset_loads=0,actual_tensor_asset_loads=0,
            original_model_data_transfer=False,actual_remote_executed=False,actual_gpu_executed=False,
            actual_source_package_built=False,actual_hydra_or_runtime_import_executed=False,
            local_static_imports_of_excluded_packages=external_local_imports,
            same_node_external5_expected_sha256=local_hashes,
            actual_same_node_external5_verified=False,
            scope='Local Git code/text byte inventory and static source-path dependency check; no dynamic import/backend/preflight proof')
        proof=HERE/'V8_ACTUAL_TRACKED_RUNTIME_DRY_SELECTION.json'
        self.assertFalse(proof.exists(),'Preserve prior dry selection; do not overwrite')
        proof.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')


if __name__=='__main__':unittest.main(verbosity=2)
