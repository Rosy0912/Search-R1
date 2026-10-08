"""Freeze the reviewed external warning-frame consumer; never deploy or edit training."""
import copy
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path

W = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('old_repair_freeze_io', W/'freeze_author200_repair_bundle.py')
io = importlib.util.module_from_spec(spec)
spec.loader.exec_module(io)
OLD = 'f5c72c9426413a18e465030efb714c9015fa84274baa3fc4e401383adc96b3fd'
PEER = '5da4808f4b7369f1b6ce1a3a0414df00b5ef4de68b43909f51a7a674974b0dcc'
SOFTWARE = '9a89a398d9e42a8390b3b225fecfc599f4953f762005d917595c995817935302'

def main():
    olddir = W/'final-acceptance-f5c72c942641'
    oldraw = io.read(olddir/'manifest.json', OLD)
    base = io.parsed(oldraw)
    peerraw = io.read(W/'INDEPENDENT_FSDP_FRAMING_SOFTWARE_REVIEW.json', PEER)
    peer = io.parsed(peerraw)
    softwareraw = io.read(W/'final-acceptance-fsdp-framing-candidate/FSDP_FRAMING_SOFTWARE_TEST_REPORT.json', SOFTWARE)
    software = io.parsed(softwareraw)
    io.need(peer['status']=='PASS_SOFTWARE_EXACT_FSDP_FRAME_REPAIR_NOT_ACTUAL75_CHECKPOINT_ACCEPTANCE'
            and peer['derived_from_manifest_sha256']==OLD
            and peer['total_independently_run_cpu_tests']==92
            and software['candidate_tools']==peer['candidate_tools'], 'Exact independent software binding differs')
    io.need(base['source_commit']==io.SOURCE and len(base['dependencies'])==268
            and len(base['tools'])==8, 'Original frozen f5 closure differs')
    tools = {n:io.read(W/'final-acceptance-fsdp-framing-candidate'/n,h) for n,h in peer['candidate_tools'].items()}
    io.need(set(tools)==set(base['tools'])
            and {n for n in tools if peer['candidate_tools'][n]!=base['tools'][n]}=={'author200_observation.py'},
            'Only the external observation helper may change')
    originals = {'manifest.json':oldraw}
    for n,h in {**base['tools'],**base['review_records']}.items():
        originals[n]=io.read(olddir/n,h)
    reviews = {n:originals[n] for n in base['review_records']}
    additions = {
        'INDEPENDENT_FSDP_FRAMING_SOFTWARE_REVIEW.json': (W/'INDEPENDENT_FSDP_FRAMING_SOFTWARE_REVIEW.json', PEER),
        'FSDP_FRAMING_SOFTWARE_TEST_REPORT.json': (W/'final-acceptance-fsdp-framing-candidate/FSDP_FRAMING_SOFTWARE_TEST_REPORT.json', SOFTWARE),
        'ACTUAL75_FSDP_WARNING_FRAMING_PROVENANCE_INDEPENDENT_REVIEW.json': (W/'ACTUAL75_FSDP_WARNING_FRAMING_PROVENANCE_INDEPENDENT_REVIEW.json', '59b4526812a5d56fd43f9b57b70971e8d6a8bbba47b2186bd6edd9f13232de52'),
        'SOURCE_FORMAL_LOG_FRAMING_TERMINAL_IMPACT_REVIEW_V2.json': (W/'SOURCE_FORMAL_LOG_FRAMING_TERMINAL_IMPACT_REVIEW_V2.json', 'd16d79903b41ab43f95315d6eceb3e84d5da5dacb4e42350631e140a5b52233e'),
        'actual75-prefix-replay-original.json': (W/'final-acceptance-fsdp-framing-candidate/actual75-prefix-replay-original.json', peer['actual_prefix_replay_sha256']),
        'FSDP_FRAMING_REPAIR_DESIGN.md': (W/'final-acceptance-fsdp-framing-candidate/FSDP_FRAMING_REPAIR_DESIGN.md', None),
    }
    for n,(p,h) in additions.items():
        io.need(n not in reviews, 'Never replace prior review member')
        reviews[n]=io.read(p,h)
    extra = {str(io.ROOT/'tools'/olddir.name/'manifest.json'):OLD}
    extra.update({str(io.ROOT/'tools'/olddir.name/n):h for n,h in base['tools'].items()})
    fsdp = peer['additional_required_dependency']
    io.need(fsdp==dict(bytes=99345,native210_already_covers_this_source=False,
        path='/mnt/data/shaoshitong/search-r1-rl/opsd/env/lib/python3.10/site-packages/torch/distributed/fsdp/fully_sharded_data_parallel.py',
        sha256='ca839d30fda12e0b12dea934e1f1bf998bd80ed716e8e4c88927691a9bb1c6a3'), 'Actual extra FSDP source differs')
    io.read(W/'actual-fsdp-warning-source-20261005T235824197037Z/fully_sharded_data_parallel.py',fsdp['sha256'])
    extra[fsdp['path']]=fsdp['sha256']
    io.need(not(set(extra)&set(base['dependencies'])) and len(extra)==10, 'Dependency union collision')
    manifest=copy.deepcopy(base)
    manifest['tools']=dict(peer['candidate_tools'])
    manifest['review_records']={n:io.digest(r) for n,r in reviews.items()}
    manifest['dependencies'].update(extra)
    io.need(len(manifest['dependencies'])==278, 'New external dependency union differs')
    manifest['limits'] += [
        'Inherited limitations retain their original historical preparation scope; no historical record is upgraded',
        'Only observation helper changes; original source188/c738/f5/native210 and all original log bytes remain frozen',
        'FSDP689 source is an additional actual external dependency, not a retroactive native210 module',
        'One exact owned FSDP warning frame is recorded separately; no hidden native report is inferred',
        'Original formal supervisor will still reject existing outer74 warning at its post-training scan; preserve real terminal failure',
        'New software/local-prefix review does not certify actual checkpoint75, wholeHF200, final51713, exit or resource release']
    raw=io.canonical(manifest);pin=io.digest(raw);out=W/('final-acceptance-'+pin[:12])
    io.need(not out.exists(), 'Never overwrite immutable new freeze')
    out.mkdir(mode=0o700)
    for n,r in {**tools,**reviews}.items():io.write(out/n,r)
    for n,h in {**manifest['tools'],**manifest['review_records']}.items():io.read(out/n,h)
    io.write(out/'manifest.json',raw)
    for p in out.iterdir():p.chmod(0o444)
    out.chmod(0o555)
    for n,r in originals.items():io.need(io.read(olddir/n)==r, 'Original f5 bytes changed')
    record=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='LOCAL_FROZEN_EXTERNAL_FSDP_FRAME_REPAIR_NOT_DEPLOYMENT_OR_CHECKPOINT_ACCEPTANCE',
        directory=str(out),remote_directory=str(io.ROOT/'tools'/out.name),
        previous_manifest_sha256=OLD,manifest_sha256=pin,tool_count=len(tools),review_count=len(reviews),
        dependency_count=278,unchanged_dependency_count=268,new_dependency_union=extra,
        changed_runtime_tools={'author200_observation.py':{'old':base['tools']['author200_observation.py'],'new':manifest['tools']['author200_observation.py']}},
        review_records=manifest['review_records'],source_commit=io.SOURCE,
        old210_unchanged=True,actual75_cpu_started=False,remote_deployment_performed=False,
        required_next='Independent exact freeze review; fresh actual278 SHA check; exclusive member publication and manifest last')
    receipt=W/('final-acceptance-freeze-'+pin[:12]+'.json');io.write(receipt,io.canonical(record))
    print(json.dumps({'directory':str(out),'manifest_sha256':pin,'receipt':str(receipt),
        'receipt_sha256':io.digest(io.read(receipt)),'members':len(list(out.iterdir())),'dependencies':278}))

if __name__=='__main__':main()
