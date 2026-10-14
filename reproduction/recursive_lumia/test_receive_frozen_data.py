import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from .receive_frozen_data import receive


class FrozenDataTransfer(unittest.TestCase):
    def fixture(self, members=None):
        payload = {'manifest.json': json.dumps({'splits': {'train': {'rows': 2}}}).encode(),
                   'raw/nq/train.jsonl': b'public test data\n'}
        spec = dict(files={k:dict(bytes=len(v),sha256=hashlib.sha256(v).hexdigest())
                           for k,v in payload.items()}, counts={'train':2},
                    manifest_sha256=hashlib.sha256(payload['manifest.json']).hexdigest())
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode='w:gz') as archive:
            for name, body in (payload.items() if members is None else members):
                member = tarfile.TarInfo(name)
                member.size = len(body)
                archive.addfile(member, io.BytesIO(body))
        stream.seek(0)
        return stream, spec, payload

    def test_complete_and_no_overwrite(self):
        stream,spec,payload = self.fixture()
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'common-data'
            result = receive(stream,spec,output,'test-source')
            self.assertEqual(result['status'],'ALL_BYTES_VERIFIED')
            for name,body in payload.items(): self.assertEqual((output/name).read_bytes(),body)
            with self.assertRaises(FileExistsError): receive(io.BytesIO(),spec,output,'other')

    def test_wrong_hash_retains_unpublished_stage(self):
        stream,spec,_ = self.fixture()
        spec['files']['raw/nq/train.jsonl']['sha256'] = '0'*64
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'common-data'
            with self.assertRaises(ValueError): receive(stream,spec,output,'test')
            self.assertFalse(output.exists())
            self.assertTrue(output.with_name('common-data.receiving').exists())

    def test_unexpected_and_missing_members_rejected(self):
        for members in [[('../outside',b'x')], []]:
            stream,spec,_ = self.fixture(members)
            with tempfile.TemporaryDirectory() as temp:
                output = Path(temp)/'common-data'
                with self.assertRaises(ValueError): receive(stream,spec,output,'test')
                self.assertFalse(output.exists())
                self.assertFalse((Path(temp)/'outside').exists())


if __name__ == '__main__':
    unittest.main()
