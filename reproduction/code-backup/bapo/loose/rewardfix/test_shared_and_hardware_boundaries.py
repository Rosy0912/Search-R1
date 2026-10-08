"""Synthetic CPU boundary counterexamples; not target hardware acceptance."""
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REMOTE = Path(__file__).parent / 'remote'
sys.path.insert(0, str(REMOTE))
import verify_shared_inputs as shared
import verify_hardware_receipt as hardware

class Boundaries(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve() / 'new'
        self.old = Path(self.tmp.name).resolve() / 'old'
        for n in ('repo','tools','logs','cache','prepared','runs','assets-root/data/paper-grpo'):
            (self.root/n).mkdir(parents=True, exist_ok=True)
        for n in ('env','retriever-gpu-lib','lib','assets-root/assets/wiki'):
            (self.old/n).mkdir(parents=True, exist_ok=True)
        for n in ('env','retriever-gpu-lib','lib','assets-root/assets'):
            (self.root/n).symlink_to(self.old/n, target_is_directory=True)
        manifest = self.root/'assets-root/assets.json'
        manifest.write_text(json.dumps([{'path':'wiki/raw', 'size':3}]))
        (self.old/'assets-root/assets/wiki/raw').write_text('raw')
        lines=[]
        for name in ('e5_Flat.index','wiki-18.jsonl'):
            p=self.old/'assets-root/assets/wiki'/name;p.write_text(name)
            lines.append(shared.digest(p)+'  '+str(p))
        derived=self.root/'logs/derived-assets.sha256';derived.write_text('\n'.join(lines)+'\n')
        self.addCleanup(patch.stopall)
        patch.object(shared,'ASSETS_SHA',shared.digest(manifest)).start()
        patch.object(shared,'DERIVED_SHA',shared.digest(derived)).start()
        paper=self.root/'assets-root/data/paper-grpo/train.parquet';paper.write_text('fixture')
        self.setup={'source_commit':shared.SOURCE, 'isolated_root':str(self.root),
            'shared_read_only_targets':{n:str((self.old/n).resolve()) for n in ('env','retriever-gpu-lib','lib')},
            'shared_assets_only':str((self.old/'assets-root/assets').resolve()),
            'copied_paper_data_files':{'data/paper-grpo/train.parquet':shared.digest(paper)}}
        (self.root/'logs/isolation-setup.json').write_text(json.dumps(self.setup))
        (self.root/'logs/shared-inputs.json').write_text(json.dumps(shared.snapshot(self.root,self.old)))

    def test_valid_isolated_fixture(self):
        shared.verify(self.root,self.old)

    def test_old_run_evidence_symlink_rejected(self):
        p=self.old/'runs/file';p.parent.mkdir();p.write_text('old')
        q=self.root/'runs/foreign';q.symlink_to(p)
        with self.assertRaises(ValueError):shared.owned(q,self.root)

    def test_owned_parent_escape_rejected(self):
        with self.assertRaises(ValueError):shared.owned(self.root/'logs/../logs/isolation-setup.json',self.root)

    def test_shared_target_old_runs_rejected(self):
        p=self.root/'assets-root/assets';p.unlink()
        (self.old/'runs').mkdir();p.symlink_to(self.old/'runs')
        with self.assertRaises(ValueError):shared.verify(self.root,self.old)

    def test_shared_record_cannot_expand_allowlist(self):
        p=self.root/'logs/shared-inputs.json';data=json.loads(p.read_text())
        data['exact_shared_targets']['runs']=str(self.old/'runs');p.write_text(json.dumps(data))
        with self.assertRaises(ValueError):shared.verify(self.root,self.old)

    def test_copied_paper_symlink_rejected(self):
        p=self.root/'assets-root/data/paper-grpo/train.parquet';p.unlink()
        q=self.old/'train';q.write_text('fixture');p.symlink_to(q)
        with self.assertRaises(ValueError):shared.verify(self.root,self.old)

    def test_copied_paper_mutation_rejected(self):
        (self.root/'assets-root/data/paper-grpo/train.parquet').write_text('changed')
        with self.assertRaises(ValueError):shared.verify(self.root,self.old)

    def test_manifest_escape_rejected(self):
        p=self.root/'assets-root/assets.json';p.write_text(json.dumps([{'path':'../logs/receipt','size':1}]))
        patch.object(shared,'ASSETS_SHA',shared.digest(p)).start()
        with self.assertRaises(ValueError):shared.snapshot(self.root,self.old)

    def test_owned_sibling_prefix_rejected(self):
        p=self.root.parent/'new-other/file';p.parent.mkdir();p.write_text('fake')
        with self.assertRaises(ValueError):shared.owned(p,self.root)

    def test_old_hardware_receipt_rejected_before_gpu_access(self):
        p=self.root/'logs/hardware-acceptance.json'
        p.write_text(json.dumps({'status':'PASS_INDEPENDENT_REVIEW','source_commit':'b20f371972bb1bc41b735c6cd350b9a779d8820d'}))
        def unexpected(*args,**kwargs):raise AssertionError('Must reject before invoking hardware')
        with self.assertRaises(ValueError):hardware.verify(self.root,command=unexpected)

    def hardware_fixture(self):
        run=self.root/'runs/bapo-hardware-123';run.mkdir()
        names=('source-commit.txt','slurm-allocation.txt','hardware.log','HARDWARE_MEASUREMENTS_RETURNED_SUCCESSFULLY',
            'retrieval-comparison.json','retrieval-cpu.json','retrieval-gpu.json','gpu-embeddings.npz',
            'parity-cpu.log','parity-gpu.log','parity-cpu-time.log','parity-gpu-time.log','cpu-process-monitor.jsonl',
            'gpu-monitor.csv','process-monitor.csv','resource-watch.jsonl','gpus-before.csv','gpus-exit.csv',
            'processes-before.csv','own-process-cleanup.json')
        for n in names:(run/n).write_text('fixture')
        (run/'source-commit.txt').write_text(shared.SOURCE)
        (run/'own-process-cleanup.json').write_text(json.dumps({'own_marker':str(run),'remaining':[]}))
        fields={'environment_sha256':'logs/environment-acceptance.json','assets_manifest_sha256':'assets-root/assets.json',
            'derived_assets_record_sha256':'logs/derived-assets.sha256','common_split_manifest_sha256':'logs/common-split-manifest.json',
            'shared_inputs_sha256':'logs/shared-inputs.json','isolation_setup_sha256':'logs/isolation-setup.json'}
        (self.root/'logs/environment-acceptance.json').write_text('{}')
        (self.root/'logs/common-split-manifest.json').write_text(json.dumps({'content_identity':{'sha256':'identity'}}))
        tools=('stage.sh','stage-gpu-inner.sh','hardware.sh','monitor.sh','resource_watch.py','owned_processes.py',
            'verify_hardware_receipt.py','verify_shared_inputs.py','flash_preflight.py','bapo-retrieval-parity.py',
            'cpu-gate.py','prepare-stage64-proof.sh')
        for n in tools:(self.root/'tools'/n).write_text('fixture')
        (self.root/'tools/hardware-launch-hashes-b09736d.json').write_text(json.dumps({n:shared.digest(self.root/'tools'/n) for n in tools}))
        review=self.root/'logs/independent-review.md';review.write_text('synthetic fixture only')
        paths=[run/n for n in names]+[self.root/n for n in fields.values()]+[self.root/'tools'/n for n in tools]+[self.root/'tools/hardware-launch-hashes-b09736d.json']
        binding=lambda p:{'path':str(p),'sha256':shared.digest(p)}
        data={'status':'PASS_INDEPENDENT_REVIEW','source_commit':shared.SOURCE,'slurm_job_id':'123',
            'run_root':str(run),'launch_stage_sha256':shared.digest(self.root/'tools/stage.sh'),
            'common_split_content_identity':'identity','gpu_uuids':['GPU-'+str(i) for i in range(8)],
            'evidence':[binding(p) for p in paths],'independent_review':binding(review)}
        data.update({k:shared.digest(self.root/n) for k,n in fields.items()})
        (self.root/'logs/hardware-acceptance.json').write_text(json.dumps(data))
        def cmd(args,**kwargs):
            if args[0]=='nvidia-smi':return '\n'.join(str(i)+', GPU-'+str(i) for i in range(8))
            return shared.SOURCE if 'rev-parse' in args else ''
        patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':','.join(str(i) for i in range(8))}).start()
        return cmd,data

    def test_hardware_valid_fixture_with_mock_uuid(self):
        cmd,_=self.hardware_fixture()
        hardware.verify(self.root,command=cmd,shared_check=lambda r:shared.verify(r,self.old))

    def test_hardware_tool_change_rejected(self):
        cmd,_=self.hardware_fixture();(self.root/'tools/monitor.sh').write_text('changed')
        with self.assertRaises(ValueError):hardware.verify(self.root,command=cmd,shared_check=lambda r:shared.verify(r,self.old))

    def test_hardware_wrong_run_rejected(self):
        cmd,data=self.hardware_fixture();data['run_root']=str(self.old/'runs/bapo-hardware-123')
        (self.root/'logs/hardware-acceptance.json').write_text(json.dumps(data))
        with self.assertRaises(ValueError):hardware.verify(self.root,command=cmd,shared_check=lambda r:shared.verify(r,self.old))

if __name__ == '__main__':
    unittest.main(verbosity=2)
