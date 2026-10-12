"""CPU classifier service for the explicitly selected live-actor probe variant.

Production rewards read current actor RPC features in ActorProbeScorer. This
service never loads an encoder or generates text; synthetic readouts admit only
the fitted classifier plumbing. Legacy frozen-encoder service stays unchanged.
"""
import argparse
import hashlib
import json
import math
import os
import threading
from pathlib import Path

from search_g1.actor_probe_contract import MODE, LAYER_CHOICES, actor_probes, validate_version
from search_g1.actor_probe_recovery import identity
from reproduction.common8.verify_data import digest

ROLE = 'actor_binary_head_service_v1'


def verify_manifest(run, manifest, *, logical_update=None):
    assert actor_probes(run)
    if logical_update is None: logical_update = manifest['round'] + 1
    validate_version(run, manifest, logical_update)
    assert manifest['parameter_ownership'] == 'independent_binary_heads_only'
    assert set(manifest['head_sha256']) == {'commit_probe.pkl','boundary_probe.pkl'}
    report = manifest['calibration_report']; proof = manifest['label_proof']
    assert report['unique_question_count'] == proof['question_count'] == 300
    assert report['question_grouped_cv'] is True and proof['group_size'] == 5
    assert report['require_all_cv_folds'] is True and report['configured_cv_folds'] == 2
    assert set(proof['question_ids']) == set(run['calibration_question_ids']) and len(set(proof['question_ids'])) == 300
    assert proof['closed_book_label'] == 'exact_match_no_retrieval'
    assert proof['reliance_label'] == 'normalized_answer_change_after_evidence_deletion'
    assert proof['model_authored_answer_spans'] is True and proof['boundary_all_reserved_questions'] is True
    assert proof['searched_valid_trajectory_count'] > 0
    assert proof['calibration_protocol']['rollout_max_new'] == 500
    assert proof['calibration_protocol']['correction_feedback'] is True
    assert proof['evidence_deletion'] == 'typed_retrieval_ranges_only_preserve_model_and_correction'
    for key in ('commit','boundary'):
        counts = report[key+'_class_counts']; auc = report[key+'_cv_auc']
        assert len(counts) == 2 and all(type(n) is int and n > 0 for n in counts)
        assert type(auc) in (int,float) and math.isfinite(auc) and 0 <= auc <= 1
    return manifest


def head_paths(base, manifest):
    result = {}
    for key,name in (('commit','commit_probe.pkl'),('boundary','boundary_probe.pkl')):
        path = Path(manifest[key])
        assert path.is_absolute() and path == path.resolve() and path.is_relative_to(base/'artifacts')
        assert path.is_file() and path.stat().st_uid == os.getuid() and path.stat().st_size <= 128*1024**2
        assert digest(path) == manifest['head_sha256'][name]
        result[key] = path
    return result


def load_heads(base, run, manifest):
    import joblib
    import numpy as np
    from reproduction.a800.recovery_binding import finite_fitted_values
    verify_manifest(run, manifest)
    result = {}
    for key,path in head_paths(base, manifest).items():
        name = key+'_probe.pkl'
        head = joblib.load(path)
        assert digest(path) == manifest['head_sha256'][name]
        assert list(head['clf'].classes_) == [0,1] and np.isfinite(head['auc']) and 0 <= head['auc'] <= 1
        assert head['layer'] in LAYER_CHOICES
        assert (head['position'],head['label']) == (('Xa','flip') if key == 'commit' else ('Xp','closedbook'))
        finite_fitted_values(vars(head['clf']))
        width = int(head['clf'].n_features_in_)
        assert width == head['clf'].n_features_in_ and 0 < width <= 65536
        result[key] = head
    return result


class HeadService:
    def __init__(self, base, run, producer):
        self.base = Path(base).resolve(); self.run = run; self.producer = producer
        assert actor_probes(run)
        self.lock = threading.RLock(); self.raw = None; self.manifest = None
        self.heads = None; self.suspended = False

    def refresh(self):
        path = self.base/'artifacts/live/manifest.json'
        assert path == path.resolve() and path.stat().st_uid == os.getuid()
        assert path.stat().st_size <= 32*1024**2
        raw = path.read_bytes()
        if raw == self.raw:
            head_paths(self.base,self.manifest)
            return
        manifest = verify_manifest(self.run,json.loads(raw))
        if self.manifest is not None:
            assert manifest['version'] != self.manifest['version'], 'Immutable Head UUID was rebound'
            assert manifest['round'] == self.manifest['round'] + 10, 'Head service skipped or repeated a refresh round'
        heads = load_heads(self.base,self.run,manifest)
        assert path.read_bytes() == raw, 'Live Head pair changed during loading'
        self.raw = raw; self.manifest = manifest; self.heads = heads; self.suspended = False

    def health(self):
        with self.lock:
            self.refresh()
            return dict(ok=True,role=ROLE,identity=identity(self.run),producer=self.producer,
                version=self.manifest['version'],round=self.manifest['round'],
                actor_step=self.manifest['actor_step'],encoder=self.manifest['encoder'],
                representation_source=MODE,head_sha256=self.manifest['head_sha256'],
                live_manifest_sha256=hashlib.sha256(self.raw).hexdigest(),
                heads_loaded=self.heads is not None,suspended=self.suspended,
                frozen_encoder_loaded=False,cuda_model_allocated_bytes=0,
                prediction_semantics='reward features come from current actor detached RPC; service smoke is classifier plumbing only')

    def suspend(self, version):
        with self.lock:
            self.refresh(); assert version == self.manifest['version']
            self.heads = None; self.suspended = True
            return dict(suspended=True,version=version,role=ROLE,
                allocated_bytes_after_release=0,frozen_encoder_loaded=False)

    def resume(self, version):
        with self.lock:
            self.refresh(); assert version == self.manifest['version']
            if self.heads is None: self.heads = load_heads(self.base,self.run,self.manifest)
            self.suspended = False
            return dict(resumed=True,version=version,role=ROLE,frozen_encoder_loaded=False)

    def smoke(self, version):
        import numpy as np
        with self.lock:
            self.refresh(); assert version == self.manifest['version'] and not self.suspended and self.heads is not None
            readouts = {}
            for key,head in self.heads.items():
                inputs = np.zeros((2,int(head['clf'].n_features_in_)),dtype=np.float32); inputs[1] = 1
                values = head['clf'].predict_proba(inputs)
                assert values.shape == (2,2) and np.isfinite(values).all()
                assert ((0 <= values) & (values <= 1)).all()
                assert np.allclose(values.sum(axis=1),1,atol=1e-6,rtol=0)
                readouts[key] = values[:,1].tolist()
            return dict(role=ROLE,identity=identity(self.run),producer=self.producer,
                version=version,round=self.manifest['round'],head_sha256=self.manifest['head_sha256'],
                readouts=readouts,synthetic_cpu_inputs=True,task_accuracy_claimed=False,
                actor_features_exercised=False,frozen_encoder_loaded=False)


def build_app(service):
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    app = FastAPI()
    class Version(BaseModel): version: str
    def call(method,version):
        try: return method(version)
        except (AssertionError,ValueError) as error: raise HTTPException(status_code=409,detail=str(error))
    @app.get('/health')
    def health(): return service.health()
    @app.post('/suspend')
    def suspend(request:Version): return call(service.suspend,request.version)
    @app.post('/resume')
    def resume(request:Version): return call(service.resume,request.version)
    @app.post('/readout_smoke')
    def smoke(request:Version): return call(service.smoke,request.version)
    return app


def main():
    import uvicorn
    from reproduction.common8.formal_contract import assert_formal_run
    from reproduction.common8.topology import validate_base
    from reproduction.a800.owned_processes import identity as process_identity
    parser = argparse.ArgumentParser(); parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--port',type=int,default=18012); args = parser.parse_args()
    base = args.base.resolve(); run = json.loads((base/'logs/run-manifest.json').read_bytes())
    validate_base(base,run); assert_formal_run(run)
    assert run['execution_backend'] == 'ssh_local' and run['training_world_size'] == 4
    assert Path(__file__).resolve().parents[2] == base/'repo'
    entry = str(Path(__file__).resolve().relative_to(base/'repo'))
    assert digest(Path(__file__)) == run['source_sha256'][entry]
    producer = process_identity(os.getpid()); producer = {k:producer[k] for k in ('pid','start_ticks','sid','uid')}
    service = HeadService(base,run,producer); service.health()
    uvicorn.run(build_app(service),host='127.0.0.1',port=args.port,workers=1)


if __name__ == '__main__': main()
