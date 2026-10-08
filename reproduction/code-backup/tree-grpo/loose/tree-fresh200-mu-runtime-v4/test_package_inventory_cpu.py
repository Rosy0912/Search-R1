"""Real fifteen-wrapper-byte packaging plus synthetic review metadata boundaries.

Only local code files are used. The tiny Git source fixture contains copied
actual API source and fake asset-shaped files; it is not full training source,
independent review, model/RPC execution or dispatch permission.
"""
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
COMMON=HERE.parent/'tree-fresh200-dispatch-preparation/fresh200_dispatch_common.py'
spec=importlib.util.spec_from_file_location('actual_fresh200_review_boundary',COMMON)
C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PackageInventoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='.package-cpu-',dir=HERE)
        self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.inventory=dict(files={name:sha(HERE/name) for name in B.WRAPPER_FILES})
        self.inventory['wrapper_files']=dict(self.inventory['files'])
        self.inputs=json.loads((HERE/'INPUTS.schema.json').read_bytes())
        self.repo=self.root/'tiny-source';self.repo.mkdir()
        actual=HERE.parent/'tree-behavior-mu-ppo'
        source_names=('verl/workers/fsdp_workers.py','search_r1/baselines/sampling_consistency.py',
                      'verl/trainer/ppo/ray_trainer.py','verl/trainer/config/ppo_tree_fresh200.yaml')
        for name in source_names:
            target=self.repo/name;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes((actual/name).read_bytes())
        self.excluded={'example/corpus.jsonl':b'{"synthetic_cpu":true}\n',
                       'example/wiki-18.jsonl.gz':b'synthetic not gzip',
                       'example/e5_Flat.index':b'synthetic index',
                       'public/test.png':b'synthetic image'}
        for name,raw in self.excluded.items():
            target=self.repo/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
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
        interface=dict(status='FRESH_CALIBRATION_MODE_AND_DIRECT_RANK_SCHEDULER_SOURCE_CPU_IMPLEMENTED',
                       source_files=source,fixture_only_not_independent_review=True)
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

    def test_positive_code_selection_excludes_asset_shapes_and_records_hashes(self):
        selection=self.manifest['source_selection']
        self.assertEqual(selection['excluded_count'],4)
        self.assertEqual(selection['excluded_files'],{name:hashlib.sha256(raw).hexdigest()
                                                      for name,raw in self.excluded.items()})
        for name in self.excluded:self.assertFalse((self.package/'source/repo'/name).exists())
        for name in ('weights.bin','model.safetensors','data.parquet','vectors.npy','wiki-18.jsonl',
                     'wiki-18.jsonl.gz','e5_Flat.index','table.csv','public/chart.png'):
            self.assertFalse(B.source_file_allowed(name),name)
        for name in ('note.py','note.yaml','note.md','receipt.json','launch.sbatch','LICENSE',
                     'verl/version/version','verl/single_controller/version/version'):
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


if __name__=='__main__':unittest.main(verbosity=2)
