"""Preselected, original-forward full log distributions; study-wide 24GiB cap."""
import fcntl
import hashlib
import json
import os
from pathlib import Path


class VocabularyDump:
    def __init__(self,study_root,output_root,identity,record):
        self.root=Path(study_root);self.output=Path(output_root);self.identity=identity;self.record=record

    def __call__(self,payload):
        import numpy as np
        self.root.mkdir(parents=True,exist_ok=True)
        arrays={k:payload[k].numpy() for k in ('student_logp','teacher_logp','outer_old_logp','target_logp') if k in payload}
        if len(arrays)!=4 or len({x.shape for x in arrays.values()})!=1 or any(x.dtype!=np.float32 or x.ndim!=1 or not np.isfinite(x).all() for x in arrays.values()):
            raise ValueError('Full-vocabulary snapshots must be finite FP32 vectors')
        prefix='token-%05d'%payload['token_index']
        ledger_path=self.root/'ledger.json'
        with (self.root/'cap.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(bytes=0,files=[],reservations=[])
            if type(ledger['bytes']) is not int or not 0<=ledger['bytes']<=24*(1<<30):
                raise ValueError('Invalid study capacity ledger')
            # Include NPY headers conservatively before writing actual files.
            expected=sum(x.nbytes+1024 for x in arrays.values())
            if ledger['bytes']+expected>24*(1<<30):
                self.record(dict(full_vocab_dump='cap_skip',token_index=payload['token_index'],identity=self.identity,study_cap_bytes=24*(1<<30)))
                return
            # Reserve first. A crash between NPY writes and the final ledger can
            # waste capacity, but cannot silently exceed the physical NPY cap.
            reservation=dict(output=str(self.output.resolve()),token_index=payload['token_index'],reserved_bytes=expected,status='reserved')
            ledger.setdefault('reservations',[]).append(reservation);ledger['bytes']+=expected
            def persist():
                temp=ledger_path.with_suffix('.tmp-%d'%os.getpid())
                with temp.open('x') as handle:
                    json.dump(ledger,handle,sort_keys=True);handle.flush();os.fsync(handle.fileno())
                os.replace(temp,ledger_path)
            persist()
            self.output.mkdir(parents=True,exist_ok=True);proofs=[]
            for name,array in arrays.items():
                path=self.output/(prefix+'-'+name+'.npy')
                with path.open('xb') as handle:
                    np.save(handle,array,allow_pickle=False);handle.flush();os.fsync(handle.fileno())
                raw=path.read_bytes()
                proof=dict(path=str(path.resolve()),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
                    dtype='float32',shape=list(array.shape),distribution=name)
                proofs.append(proof)
            actual=sum(x['bytes'] for x in proofs)
            if actual>expected:raise ValueError('NPY header exceeded conservative reservation')
            ledger['bytes']+=actual-expected;ledger['files'].extend(proofs)
            reservation.update(status='complete',actual_bytes=actual,files=proofs);persist()
            self.record(dict(full_vocab_dump='complete',token_index=payload['token_index'],files=proofs,
                identity=self.identity,scope='original forward GPU log_softmax detached CPU FP32; no backward rewrite',
                probabilities_persisted=False,probabilities_reconstructible_by_exp=True,study_used_bytes=ledger['bytes']))
