"""CPU metadata fixtures; fdinfo strings include an actual h100-9 synthetic sample."""
import copy
import hashlib
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import resource_adapter_v2 as A
import test_resource_adapter as base
base.A=A


class V2ResourceTests(base.ResourceTests):
    def test_actual_h1009_fdinfo_parent_owner_and_foreign_holder_distinguished(self):
        from pathlib import Path
        raw=json.loads((Path(__file__).resolve().parents[2]/'outputs/tree-instruct-300/PPO_PROBE_FDINFO_CPU_RAW.json').read_text())
        st=SimpleNamespace(st_dev=os.makedev(raw['device_major'],raw['device_minor']),st_ino=raw['inode'])
        with patch.object(A.os,'major',return_value=raw['device_major']),patch.object(A.os,'minor',return_value=raw['device_minor']):
            self.assertTrue(A.fdinfo_exclusive_owner(raw['before'],st))
            with self.assertRaisesRegex(ValueError,'no unique kernel FLOCK'):A.fdinfo_exclusive_owner(raw['after'],st)

    def test_read_posix_partial_wrong_inode_or_malformed_lock_refused(self):
        st=SimpleNamespace(st_dev=os.makedev(253,1),st_ino=123)
        text='lock:\t1: FLOCK ADVISORY WRITE 42 fd:01:123 0 EOF\n'
        with patch.object(A.os,'major',return_value=253),patch.object(A.os,'minor',return_value=1):
            self.assertTrue(A.fdinfo_exclusive_owner(text,st))
        for changed in [text.replace('WRITE','READ'),text.replace('FLOCK','POSIX'),
                        text.replace('fd:01:123','fd:01:124'),text.replace('0 EOF','2 EOF'),
                        text.replace('42','0'),text+text,'','lock: 1: garbage']:
            with patch.object(A.os,'major',return_value=253),patch.object(A.os,'minor',return_value=1),self.assertRaises(ValueError):A.fdinfo_exclusive_owner(changed,st)

    def test_strict_bound_json_rejects_duplicates_constants_and_exponent_overflow(self):
        self.assertEqual(A.strict_json_bytes(b'{"ok":1}'),{'ok':1})
        for text in [b'{"x":1,"x":2}',b'{"x":NaN}',b'{"x":Infinity}',b'{"x":1e999}']:
            with self.assertRaises(ValueError):A.strict_json_bytes(text)

    def test_normalized_release_binds_every_bundle_payload_field(self):
        b={'output':str(A.ROOT/'runs/tree-probability-probe-99999/gpu-records'),
           'status':'GPU_PROBE_BUNDLE_INDEPENDENTLY_RELEASED','config_text':'mini: 384',
           'release':{'path':'a','sha256':'b'}}
        template={k:v for k,v in b.items() if k!='release'};template['output']='@OWN_RUN_OUTPUT@'
        expected=hashlib.sha256(json.dumps(template,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
        self.assertEqual(A.bundle_payload_sha256(b,'99999'),expected)
        altered=copy.deepcopy(b);altered['config_text']='mini: 192'
        self.assertNotEqual(A.bundle_payload_sha256(altered,'99999'),expected)
        altered['output']=str(A.ROOT/'runs/tree-probability-probe-88888/gpu-records')
        with self.assertRaises(ValueError):A.bundle_payload_sha256(altered,'99999')

    def test_installed_identity_empty_version_or_actual_module_path_refused(self):
        with self.assertRaisesRegex(ValueError,'source set'):A.installed_bindings([],A.INSTALLED_VERSIONS)
        entries=[dict(module=n,path=str(A.ROOT/'env/lib/python3.10/site-packages/source.py'),sha256='0'*64) for n in A.INSTALLED_MODULES]
        with self.assertRaisesRegex(ValueError,'version identity'):A.installed_bindings(entries,{})
        with patch.object(A.importlib.metadata,'version',side_effect=lambda n:A.INSTALLED_VERSIONS[n]),\
                patch.object(A.importlib,'import_module',return_value=SimpleNamespace(__file__='/wrong/source.py')):
            with self.assertRaisesRegex(ValueError,'imported implementation'):A.installed_bindings(entries,A.INSTALLED_VERSIONS)


if __name__=='__main__':unittest.main(verbosity=2)
