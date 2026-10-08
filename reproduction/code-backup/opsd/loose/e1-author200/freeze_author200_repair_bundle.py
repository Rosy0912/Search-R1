"""Local-only immutable tool freeze with exact failed25 lineage; no deployment."""
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

WORK = Path(__file__).resolve().parent
ROOT = Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct200-author-20261006')
RUN = ROOT / 'runs/formal-3261'
OLD_SHA = '8441e10fa4565bf53f7baf24a57a60ddbe9844dc979a2c9a0c97250219df7019'
SOURCE = '188f4e5e7b63cf479290d0888a4a075499886750'
TOOLS = {
    'author200_observation.py': '29592450920eb6bc4f556138f4d0a73757bf3fc05703f7860c3b2f69c02c0295',
    'check_author200_evaluations.py': '10483cf0140718f3755ac98de7c0424ac4e593463a63378e8a7ea5073fb69b41',
    'checkpoint_cpu_pipeline.py': 'ef6dedc6dfde43c836dd96236d71b280d0c4adf9e53233eb86434d5f8463097a',
    'checkpoint_overlap.py': '20c9fb1b720bd345f92b3cdfca0b4b19dc6d260e84c2d16429af497f8de45062',
    'checkpoint_repair_cpu.py': 'a740a5106b7b8c338da294d14cd6437a711ac79e74b50b28770ed9814b5a60e2',
    'hf_author200_cpu.py': '9fb84418711bfd9a74c786448d625cad7a5dc723e948932e6e908f60e6810931',
    'inspect_author200_checkpoint_cpu.py': 'f7d8482e54c727c878a3114d4d748889e4bcd38cc7dac573000f6ba2cda1864a',
    'observe_author200_checkpoint.py': '77f7099ee028ef2cc2b8fa0875f00940663265ec2285337e7900a5b946c38f7a',
}
REVIEWS = {
    'SAMPLER_SEED_REPAIR_INDEPENDENT_REVIEW.json': ('SAMPLER_SEED_REPAIR_INDEPENDENT_REVIEW.json', '6ee7c32a0f03c1ac0b21fd711613e119944016dde7a270eb430ac1b118a6afa2'),
    'ACTUAL_SMALL_DRIVER_SAMPLER_REPAIR_JOIN.json': ('final-acceptance-candidate/ACTUAL_SMALL_DRIVER_SAMPLER_REPAIR_JOIN.json', '9d0d74cbc62179b32249ce29745a55f33c7e1d534e8dc86548cb59cdde7e500f'),
    'ACTUAL25_CPU_FAILURE_INDEPENDENT_REVIEW.json': ('ACTUAL25_CPU_FAILURE_INDEPENDENT_REVIEW.json', 'a211a37c6485f7c772eed21e95892929ebd61b2e73504683b995cfa192928b2f'),
    'CHECKPOINT_REPAIR_RUNNER_INDEPENDENT_REVIEW.json': ('CHECKPOINT_REPAIR_RUNNER_INDEPENDENT_REVIEW.json', '9a1dfdf0cd0264a549a0707a40c57b27075bd070dc24c502073f80ea25f9d5ae'),
    'CHECKPOINT_REPAIR_RUNNER_LOCAL_CPU_SOFTWARE_TEST_REPORT.json': ('checkpoint-repair-cpu-tests-v1/software-test-report.json', 'cdc01c4fe01161476383189e95beba3f2f35081f500bf3e867d975ad1f9659f3'),
    'SAMPLER_CHECKER_CORRECTION_ADDENDUM.md': ('SAMPLER_CHECKER_CORRECTION_ADDENDUM.md', 'eb93d57d12d259de9c25ba4e2e555a8702a2e60558875d95fd0848af37776ce2'),
}
FAILURES = {
    'native-exit-original.json': 'acf00633f70bcf95c11e5635df9724235531121177740d81e73768e692a86e46',
    'srun-exit-original.json': 'faec00c6c152b566416d818c2de3fe5b629e27ef4920875ef8f688026c799a37',
    'controller-exit-original.json': '1fbf8672f6873a68c693c6dfcf0ea1263030f5d6ccfdca16bf12e7766c91694a',
}


def need(ok, message):
    if not ok: raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode() + b'\n'


def parsed(raw):
    def pairs(values):
        out = {}
        for key,value in values:
            need(key not in out, 'Duplicate JSON key')
            out[key] = value
        return out
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: need(False, 'Nonfinite JSON'))


def read(path, pin=None):
    p = Path(path)
    need(p.is_absolute() and p.resolve(strict=True)==p and p.is_file()
         and p.stat().st_uid==os.getuid(), 'Owned physical local input required')
    before = p.stat()
    with p.open('rb') as f:
        opened=os.fstat(f.fileno());raw=f.read();closed=os.fstat(f.fileno())
    after=p.stat()
    fields=('st_dev','st_ino','st_size','st_uid','st_mtime_ns','st_ctime_ns')
    need(tuple(getattr(before,k) for k in fields)==tuple(getattr(opened,k) for k in fields)
         ==tuple(getattr(closed,k) for k in fields)==tuple(getattr(after,k) for k in fields),
         'Local input changed while reading')
    if pin is not None:need(digest(raw)==pin,'Local input SHA differs: '+str(p))
    return raw


def write(path, raw):
    with Path(path).open('xb') as f:
        f.write(raw);f.flush();os.fsync(f.fileno())


def main():
    old=WORK/'final-acceptance-8441e10fa456'
    old_manifest=read(old/'manifest.json',OLD_SHA);base=parsed(old_manifest)
    need(base['source_commit']==SOURCE and len(base['tools'])==7 and len(base['dependencies'])==254,
         'Actual original8441 closure differs')
    old_inputs={old/'manifest.json':old_manifest}
    for name,pin in base['tools'].items():old_inputs[old/name]=read(old/name,pin)
    for name,pin in base['review_records'].items():old_inputs[old/name]=read(old/name,pin)
    inherited=parsed(read(WORK/'final-acceptance-deployment-8441e10fa456-originals/before-dependency-check-original.json',
        'c63877fac563eff5f138916bf5d5a2bbb9f9ed221f441cc98a3085a0762e2cd1'))
    need(inherited['source_commit']==SOURCE and inherited['source_clean'] is True
         and {r['path']:r['sha256'] for r in inherited['dependencies']}==base['dependencies'],
         'Original remote dependency proof differs from inherited254')
    repo=WORK.parent/'opsd-e1-300'
    need(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==SOURCE
         and not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True),
         'Local frozen source188 must remain clean')
    source_prefix=str(ROOT/'repo-188f4e5e7b63')+'/'
    local_source_count=0
    for path,pin in base['dependencies'].items():
        if path.startswith(source_prefix):read(repo/path[len(source_prefix):],pin);local_source_count+=1
    tools={name:read(WORK/'final-acceptance-candidate'/name,pin) for name,pin in TOOLS.items()}
    need({name for name in base['tools'] if base['tools'][name]!=TOOLS[name]}=={'inspect_author200_checkpoint_cpu.py'},
         'Unexpected changed original runtime tool')
    reviews={name:old_inputs[old/name] for name in base['review_records']}
    review_pins=dict(base['review_records'])
    for name,(relative,pin) in REVIEWS.items():reviews[name]=read(WORK/relative,pin);review_pins[name]=pin
    seed=parsed(reviews['SAMPLER_SEED_REPAIR_INDEPENDENT_REVIEW.json'])
    need(seed['repaired_native_inspector_sha256']==TOOLS['inspect_author200_checkpoint_cpu.py']
         and seed['unchanged_hf_inspector_sha256']==TOOLS['hf_author200_cpu.py'], 'Repaired native/HF review differs')
    join=parsed(reviews['ACTUAL_SMALL_DRIVER_SAMPLER_REPAIR_JOIN.json'])
    need(join['new_checker_sha256']==TOOLS['inspect_author200_checkpoint_cpu.py']
         and join['actual_sampler_seed_5'] is None and join['actual_sampler_seed_25'] is None
         and join['full_native_checkpoint_accepted'] is False, 'Actual small join scope differs')
    peer=parsed(reviews['CHECKPOINT_REPAIR_RUNNER_INDEPENDENT_REVIEW.json'])
    need(peer['runner_sha256']==TOOLS['checkpoint_repair_cpu.py']
         and peer['new_native_inspector_sha256']==TOOLS['inspect_author200_checkpoint_cpu.py']
         and peer['unchanged_launcher_sha256']==TOOLS['checkpoint_overlap.py']
         and peer['unchanged_pipeline_sha256']==TOOLS['checkpoint_cpu_pipeline.py'], 'Runner review runtime bytes differ')
    software=parsed(reviews['CHECKPOINT_REPAIR_RUNNER_LOCAL_CPU_SOFTWARE_TEST_REPORT.json'])
    need(software['runner_source_sha256_before']==software['runner_source_sha256_after']==TOOLS['checkpoint_repair_cpu.py']
         and software['passed_software_only'] is True and software['actual_local_suite_exit_code']==0,
         'Local software report differs')
    failure=parsed(reviews['ACTUAL25_CPU_FAILURE_INDEPENDENT_REVIEW.json'])
    need(failure['failed_tools_manifest_sha256']==OLD_SHA and failure['checkpoint_content_pass'] is False
         and failure['original_exits']=={name:1 for name in FAILURES}, 'Original failure review differs')
    snap=WORK/'actual-snapshots/3261-20261005T221857135211Z/original-files'
    marker=read(snap/'evidence/checkpoint-cpu-25-launch.json',
        'feac6e0c56a35e7725307b0d65c6f0437a0ae18bba13bea07e774721b1a2e88f')
    inventory=read(snap/'evidence/checkpoint-25-cpu-inspection-once-v1/observation/observed-checkpoint-inventory.json',
        '32f2c38c121b0c59ab26036c358b4fd749218f8e3980d6700f54c78fabbe6c4e')
    need(parsed(marker)['checkpoint_step']==25 and parsed(marker)['tools_manifest_sha256']==OLD_SHA,
         'Actual old marker differs')
    need(parsed(inventory)['kind']=='OBSERVED_AUTHOR200_CHECKPOINT_INVENTORY_NOT_TRAINING_MANIFEST'
         and parsed(inventory)['checkpoint_step']==25, 'Actual observed25 inventory scope differs')
    exit_wire_path=WORK/'checkpoint25-native-status-20261005T222408Z-original.json'
    exit_wire=parsed(read(exit_wire_path,'be1d5d015824e77148d466be973d9299bd16e19322ff65bf3c5540e7da80cf49'))
    stream_wire_path=WORK/'checkpoint25-cpu-failure-streams-original.json'
    stream_wire=parsed(read(stream_wire_path,'935ffb6f4c24f21e6841c11f56d6499a0a4aaad2f4f3f022fe11519ff3a51202'))
    original_output=RUN/'evidence/checkpoint-25-cpu-inspection-once-v1'
    originals={RUN/'evidence/checkpoint-cpu-25-launch.json':marker,
               original_output/'observation/observed-checkpoint-inventory.json':inventory}
    recovered=[]
    recovery=WORK/'repair25-lineage-exact-recovered-v1'
    need(not recovery.exists(),'Do not overwrite recovered lineage')
    recovery.mkdir(mode=0o700)
    for name,pin in FAILURES.items():
        record=exit_wire['files'][name];value=record['content'];raw=canonical(value)
        need(type(value['returncode']) is int and value['returncode']==1
             and record['sha256']==pin and record['bytes']==len(raw) and digest(raw)==pin,
             'Cannot recover exact original failed exit bytes')
        write(recovery/name,raw);originals[original_output/name]=raw
        recovered.append(dict(remote_path=str(original_output/name),local_path=str(recovery/name),
                              bytes=len(raw),sha256=pin,wire_path=str(exit_wire_path),
                              recovery_scope='Exact bytes proved by original recorded SHA/length, not fabricated status'))
    record=stream_wire['files']['native-stderr-original.txt'];raw=record['content'].encode('utf-8')
    need(len(raw)==record['bytes'] and digest(raw)==record['sha256']
         =='43277dcc82f583862054290d83e2c608e497ce4815c5d4c5586600652ccc20f8'
         and raw.endswith(b'ValueError: Sampler position/config differs: seed\n'),
         'Original failed native stderr bytes differ')
    write(recovery/'native-stderr-original.txt',raw);originals[original_output/'native-stderr-original.txt']=raw
    recovered.append(dict(remote_path=str(original_output/'native-stderr-original.txt'),
        local_path=str(recovery/'native-stderr-original.txt'),bytes=len(raw),sha256=digest(raw),
        wire_path=str(stream_wire_path),recovery_scope='Exact UTF-8 original stream from recorded wire and SHA'))
    dependencies=copy.deepcopy(base['dependencies'])
    extra={str(ROOT/'tools/final-acceptance-8441e10fa456'/'manifest.json'):OLD_SHA}
    for name,pin in base['tools'].items():extra[str(ROOT/'tools/final-acceptance-8441e10fa456'/name)]=pin
    for path,raw in originals.items():extra[str(path)]=digest(raw)
    need(not (set(extra)&set(dependencies)) and len(extra)==14, 'Unexpected old closure conflict/alias')
    dependencies.update(extra)
    need(len(dependencies)==268 and all(type(p) is str and Path(p).is_absolute()
         and re.fullmatch('[0-9a-f]{64}',h) for p,h in dependencies.items()), 'Invalid dependency union')
    manifest=copy.deepcopy(base);manifest['tools']=dict(TOOLS);manifest['dependencies']=dependencies
    manifest['review_records']=review_pins
    manifest['limits']=list(base['limits'])+[
        'Original8441 inspection failed; native/srun/controller exit1 is immutable historical evidence',
        'Sampler seed is exact frozen constructor defaultNone; model baseline seed42 is distinct',
        'Repair25 runner only uses a fresh fixed repair-v2 marker/output; no training control or acceptance signature',
        'New review records qualify software/small-state join only; actual native2.4 repaired25 remains unaccepted']
    raw_manifest=canonical(manifest);pin=digest(raw_manifest);folder=WORK/('final-acceptance-'+pin[:12])
    need(not folder.exists(), 'Never overwrite an immutable frozen bundle')
    folder.mkdir(mode=0o700)
    for name,raw in {**tools,**reviews}.items():
        need(Path(name).name==name,'Unsafe bundle member');write(folder/name,raw)
    for name,pin0 in {**TOOLS,**review_pins}.items():read(folder/name,pin0)
    # Manifest is deliberately the last bundle member created.
    write(folder/'manifest.json',raw_manifest);read(folder/'manifest.json',pin)
    for p in folder.iterdir():p.chmod(0o444)
    folder.chmod(0o555)
    for p,raw0 in old_inputs.items():need(read(p)==raw0,'Original8441 input changed')
    record=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='LOCAL_FROZEN_REPAIR25_EXTERNAL_TOOLS_NOT_DEPLOYMENT_OR_CHECKPOINT_PASS',
        directory=str(folder),remote_directory=str(ROOT/'tools'/folder.name),
        manifest_sha256=pin,tool_count=8,dependency_count=268,
        unchanged_original_dependency_count=254,new_dependency_union=extra,
        local_source_files_read_again=local_source_count,
        inherited_remote_original_proof=dict(path=str(WORK/'final-acceptance-deployment-8441e10fa456-originals/before-dependency-check-original.json'),
            sha256='c63877fac563eff5f138916bf5d5a2bbb9f9ed221f441cc98a3085a0762e2cd1',
            utc=inherited['utc'],scope='Historical actual remote254 verification; not a new remote read'),
        review_records=review_pins,recovered_exact_lineage=recovered,
        changed_runtime_tools={'inspect_author200_checkpoint_cpu.py':dict(old=base['tools']['inspect_author200_checkpoint_cpu.py'],new=TOOLS['inspect_author200_checkpoint_cpu.py']),
                               'checkpoint_repair_cpu.py':dict(old=None,new=TOOLS['checkpoint_repair_cpu.py'])},
        previous_manifest_sha256=OLD_SHA,remote_deployment_performed=False,
        actual_checkpoint25_content_acceptance=False,
        required_next='Independent freeze/dependency review and root fresh actual remote268 check; manifest publish last')
    receipt=WORK/('final-acceptance-freeze-'+pin[:12]+'.json');write(receipt,canonical(record))
    print(canonical(dict(receipt=str(receipt),receipt_sha256=digest(read(receipt)),
                         directory=str(folder),manifest_sha256=pin,dependency_count=268)).decode())


if __name__=='__main__':main()
