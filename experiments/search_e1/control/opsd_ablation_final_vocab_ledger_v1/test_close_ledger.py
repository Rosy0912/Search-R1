"""Tiny synthetic accounting checks; these are not real-array/70-writer audits."""
import copy
from pathlib import Path
import stat
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parent))
from close_ledger import CAP, DISTRIBUTIONS, reconcile, writer_closure


def example():
    root = '/fixed/group-C0/runs'
    output = root+'/outers/outer-00000/training/attempt-fixed/rank-0/vocabulary-update-00000001-retry-0000-pair-000'
    files = []
    physical = []
    for i, distribution in enumerate(sorted(DISTRIBUTIONS)):
        # teacher and target equal bytes/SHA, but still two separate files.
        sha = ('a' if distribution in ('teacher_logp','target_logp') else 'b' if distribution=='student_logp' else 'c')*64
        ref = dict(path=output+'/token-00000-'+distribution+'.npy', sha256=sha, bytes=607872,
                   dtype='float32',shape=[151936],distribution=distribution)
        files.append(ref)
        physical.append(dict(ref,device=1,inode=i+10,mode=stat.S_IFREG|0o600,nlink=1))
    n = sum(f['bytes'] for f in files)
    reservation = dict(output=output,token_index=0,status='complete',reserved_bytes=4*(151936*4+1024),actual_bytes=n,files=files)
    ledger = dict(bytes=n,files=files,reservations=[reservation])
    committed = [dict(outer=0,token_index=0,record=dict(full_vocab_dump='complete',files=files))]
    keys = {(f['path'],f['sha256'],f['bytes']) for f in files}
    return ledger,physical,committed,keys,keys,{},[root]


class Accounting(unittest.TestCase):
    def test_complete_and_equal_content_is_not_deduplicated(self):
        result = reconcile(*example())
        self.assertTrue(result['artifact_complete'])
        self.assertEqual(result['physical_unique_array_files'],4)
        self.assertEqual(result['physical_unique_NPY_bytes'],2431488)

    def test_capacity_duplicate_unregistered_and_wrong_join_reject(self):
        for kind in ('capacity','duplicate','unregistered','join'):
            with self.subTest(kind=kind):
                args = copy.deepcopy(example())
                if kind=='capacity': args[0]['bytes']=CAP+1
                elif kind=='duplicate': args[0]['reservations'].append(copy.deepcopy(args[0]['reservations'][0]))
                elif kind=='unregistered': args[1][0]['path']=args[1][0]['path'].replace('token-00000','token-00001')
                else:
                    args[2][0]['record']['files']=copy.deepcopy(args[2][0]['record']['files'])
                    args[2][0]['record']['files'][0]['sha256']='d'*64
                with self.assertRaises(ValueError): reconcile(*args)

    def test_missing_persistence_or_reservation_and_fake_quiescence_stay_partial(self):
        args = copy.deepcopy(example()); args[3].clear()
        self.assertFalse(reconcile(*args)['artifact_complete'])
        args = list(copy.deepcopy(example())); r=args[0]['reservations'][0]
        r['status']='reserved';r.pop('files');r.pop('actual_bytes');args[0]['files']=[]
        args[0]['bytes']=r['reserved_bytes'];args[2]=[];args[1]=[]
        self.assertFalse(reconcile(*args)['artifact_complete'])
        # A manifest Boolean cannot bypass reading seven real native terminals.
        class Docs:
            def read(self, ref): return dict(schema='opsd-ablation-config-v1',variant='C0',run_id='r',
                scientific=dict(outer_iterations=10),paths=dict(output_root='/fixed/group-C0/runs'),precision=dict(mode='bf16'))
        with self.assertRaises((ValueError,KeyError)):
            writer_closure(dict(variant='C0',config=dict(path='/config',sha256='a'*64,bytes=1),writer_quiescent=True),
                           dict(outers=[],configs=[]),Docs())


if __name__=='__main__': unittest.main()
