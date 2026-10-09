#!/usr/bin/env python3
"""Local, immutable-backup-only aggregation of explicitly committed seven-group token logs."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import time

VARIANTS = ('C0','C1','M','R','W','P','J')
import gzip
import tarfile
MASS_KEYS = {k: f'clipped_{k}_probability_mass' for k in ('student', 'teacher', 'mixture')}
LOSS_KEYS = {'raw': 'pre_clip_full_vocabulary_kl', 'post': 'post_clip_full_vocabulary_loss'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def metadata(s):
    return dict(bytes=s.st_size, device=s.st_dev, inode=s.st_ino, mode=s.st_mode,
                mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def stable_json(path, expected_sha=None):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts and all(not q.is_symlink() for q in (path,*path.parents)) and stat.S_ISREG(path.stat().st_mode), 'Canonical nonsymlink regular JSON required')
    before = metadata(path.stat())
    with path.open('rb') as f:
        require(metadata(os.fstat(f.fileno())) == before, 'JSON opened identity changed')
        raw = f.read()
        require(metadata(os.fstat(f.fileno())) == before, 'JSON changed during read')
    require(metadata(path.stat()) == before, 'JSON path changed during read')
    digest = hashlib.sha256(raw).hexdigest()
    require(expected_sha is None or digest == expected_sha, 'JSON SHA mismatch: ' + str(path))
    return json.loads(raw), dict(path=str(path.resolve()), sha256=digest, **before)


def finite(value, field):
    require(type(value) in (int, float) and math.isfinite(value), 'non-finite/non-numeric: ' + field)
    return float(value)


def equal_fp32_log(a, b):
    # Logged token scalars originate from FP32 tensors. Native recorded scalar
    # summaries sum Python lists chunk by chunk; this local sum is sequential.
    # Neither is a reconstruction of the differentiable loss/GEMM/reduction.
    return math.isclose(a, b, rel_tol=2e-6, abs_tol=2e-6)


class FixedBackups:
    """Explicit accepted direct or gzip backup inputs; no extraction or Node I/O."""
    def __init__(self, specifications, run):
        self.files={}; self.inputs=[]; self.reads=[]; self.seen=set(); self.archives=[]; self.active=None
        for spec in specifications:
            kind=spec['kind']
            if kind=='direct':
                source,sp=stable_json(spec['source']['path'],spec['source']['sha256'])
                local,lp=stable_json(spec['local']['path'],spec['local']['sha256'])
                require(source['accepted'] is True and source['all_source_pinned_metadata_stable'] is True and source['run_id']==run,'Direct source unaccepted/wrong run')
                require(local['accepted'] is True and local['exact_inventory'] is True and local['local_metadata_stable'] is True and local['no_symlinks_or_special_files'] is True and local['source_sha_matched_all_files'] is True,'Direct local proof unaccepted')
                sm={r['relative']:r for r in source['files']};lm={r['relative']:r for r in local['files']}
                require(len(sm)==len(source['files'])==source['file_count'] and len(lm)==len(local['files'])==local['file_count'] and sm.keys()==lm.keys(),'Direct inventory differs')
                require(sum(r['bytes'] for r in sm.values())==source['total_bytes']==local['total_bytes'],'Direct bytes differ')
                destination=Path(local['destination']).absolute()
                for name,r in sm.items():
                    l=lm[name];p=Path(l['path']).absolute()
                    require(r['sha256']==l['sha256'] and r['bytes']==l['bytes'] and p==destination/name and str(Path(source['source_root'])/name)==r['path'],'Direct member identity differs')
                    self.add(r,dict(kind=kind,path=p,metadata=l['metadata']))
                self.inputs.append(dict(kind=kind,source=sp,local=lp))
            elif kind=='gzip':
                inv,ip=stable_json(spec['inventory']['path'],spec['inventory']['sha256'])
                proof,pp=stable_json(spec['complete']['path'],spec['complete']['sha256'])
                require(proof['accepted'] is True and proof['run_id']==inv['run_id']==run and proof['inventory_sha256']==ip['sha256'] and proof['actual_ssh_returncode']==0,'Gzip proof/run/inventory differs')
                require(all(proof[k] is True for k in ('all_local_decompressed_member_SHA_match','gzip_footer_and_terminal_padding_verified','contiguous_regular_tar_members_verified','source_noncache_file_set_stable_after','source_stream_fixed_handle_SHA')),'Gzip original acceptance absent')
                members={r['relative']:r for r in proof['members']}; rows={r['relative']:r for r in inv['files']}
                require(len(members)==len(proof['members'])==len(rows)==len(inv['files'])==proof['files']==inv['expected_file_count'] and members.keys()==rows.keys(),'Gzip complete inventory differs')
                require(sum(r['bytes'] for r in rows.values())==inv['bytes']==proof['member_bytes'],'Gzip bytes differ')
                ap=proof['archive'];p=Path(ap['path']).absolute();self.path_check(p)
                actual=metadata(p.stat());require(all(actual[k]==ap[k] for k in actual),'Accepted archive metadata changed')
                arc=dict(path=p,metadata=actual,proof=ap,members=rows);self.archives.append(arc)
                for name,r in rows.items():
                    require(members[name]['sha256']==r['sha256'] and members[name]['bytes']==r['bytes'] and str(Path(inv['source'])/name)==r['path'],'Gzip member identity differs')
                    self.add(r,dict(kind=kind,archive=arc,name=name))
                self.inputs.append(dict(kind=kind,inventory=ip,complete=pp,archive_SHA_origin='Original complete acceptance; current metadata continuity, not a new whole-archive SHA'))
            else:raise ValueError('Explicit direct/gzip backup kind required')

    @staticmethod
    def path_check(path):
        require(path.is_absolute() and '..' not in path.parts,'Canonical path required')
        for current in (path,*path.parents):require(not current.is_symlink(),'Symlink member/ancestor refused')
        require(stat.S_ISREG(path.stat().st_mode),'Regular member required')

    def add(self,row,local):
        require(row['path'] not in self.files,'Duplicate source across backup scopes')
        relative=Path(row['relative'])
        require(not relative.is_absolute() and '..' not in relative.parts,'Unsafe backup member')
        self.files[row['path']]=(row,local)

    def opened(self,ref):
        require(ref['path'] in self.files and ref['path'] not in self.seen,'Missing/duplicate committed reference')
        source,local=self.files[ref['path']]
        require(ref['sha256']==source['sha256'] and ref['bytes']==source['bytes'],'Receipt/backup SHA or bytes differ')
        recorded=source.get('pinned_metadata',source)
        for k in ('device','inode','bytes','mtime_ns','ctime_ns','mode'):
            if k in ref:require(ref[k]==recorded[k],'Committed source metadata differs: '+k)
        if local['kind']=='direct':
            p=local['path'];self.path_check(p);require(metadata(p.stat())==local['metadata'],'Direct member changed')
            f=p.open('rb');require(metadata(os.fstat(f.fileno()))==local['metadata'],'Opened direct member differs')
        else:
            require(self.active is not None and self.active[0]==ref['path'],'Compressed member must be processed in sequential visit')
            f=self.active[1];p=local['archive']['path']
        return f,source,local,p

    def finish(self,ref,f,source,local,path,digest,bytes_read,category):
        require(digest==ref['sha256'] and bytes_read==ref['bytes'],'Fresh selected member SHA/bytes differs')
        if local['kind']=='direct':require(metadata(os.fstat(f.fileno()))==local['metadata']==metadata(path.stat()),'Direct member changed during read')
        else:require(metadata(path.stat())==local['archive']['metadata'],'Archive changed during selected read')
        self.seen.add(ref['path']);self.reads.append(dict(source_path=ref['path'],local_path=str(path),sha256=digest,bytes=bytes_read,category=category,kind=local['kind']))

    def receipt(self,ref):
        f,s,l,p=self.opened(ref)
        with f:
            raw=f.read();self.finish(ref,f,s,l,p,hashlib.sha256(raw).hexdigest(),len(raw),'committed_update_receipt')
        return json.loads(raw)

    def visit(self,references,callback):
        refs={r['path']:r for r in references}
        require(len(refs)==len(references),'Duplicate requested visit')
        require(set(refs)<=set(self.files),'Committed files missing from accepted backups')
        done=set()
        for path,ref in refs.items():
            if self.files[path][1]['kind']=='direct':callback(ref);done.add(path)
        for arc in self.archives:
            wanted={path:ref for path,ref in refs.items() if self.files[path][1].get('archive') is arc}
            if not wanted:continue
            seen=set();p=arc['path'];self.path_check(p)
            require(metadata(p.stat())==arc['metadata'],'Archive changed before visit')
            with p.open('rb') as raw:
                require(metadata(os.fstat(raw.fileno()))==arc['metadata'],'Archive opened identity differs')
                with gzip.GzipFile(fileobj=raw,mode='rb') as gz:
                    with tarfile.open(fileobj=gz,mode='r|') as tar:
                        for member in tar:
                            require(member.isfile() and member.name in arc['members'] and member.name not in seen,'Unexpected/duplicate archived member')
                            row=arc['members'][member.name];require(member.size==row['bytes'],'Archived member size differs')
                            seen.add(member.name)
                            if row['path'] in wanted:
                                self.active=(row['path'],tar.extractfile(member))
                                try:callback(wanted[row['path']]);done.add(row['path'])
                                finally:self.active=None
                    # Read to physical gzip EOF for CRC; original strict footer proof remains bound.
                    while gz.read(4<<20):pass
                require(metadata(os.fstat(raw.fileno()))==arc['metadata'],'Archive changed during visit')
            require(seen==set(arc['members']) and metadata(p.stat())==arc['metadata'],'Archive cover/metadata differs')
        require(done==set(refs),'Requested committed members not processed')


def fresh_stats():
    return dict(valid_token_rows=0, clipped_count=0, hit_tokens=0, vocabulary_slots=0,
                auxiliary_vocabulary_dump_rows=0,
                loss_sum={k: 0. for k in LOSS_KEYS}, mass_sum={k: 0. for k in MASS_KEYS},
                mass_max={k: 0. for k in MASS_KEYS}, vocabulary_sizes=set())


def token_log(backups, ref, pair, receipt, rank):
    f, s, l, path = backups.opened(ref)
    digest = hashlib.sha256()
    size = 0
    stats = fresh_stats()
    with f:
        line = f.readline()
        digest.update(line); size += len(line)
        require(line.endswith(b'\n'), 'incomplete token header')
        header = json.loads(line)
        require(header['schema'] == 'opsd-online-training-token-header-v1', 'token header schema differs')
        require(header['run_id'] == receipt['run_id'] and header['outer_index'] == receipt['outer_index'] and
                header['optimizer_update'] == receipt['optimizer_updates'] and header['rank'] == rank and
                header['pair_index'] == pair['pair_index'], 'header identity differs')
        require(header['contract'] == receipt['contract'] and header['component_clip'] == 10 and
                header['pair_weight'] == pair['pair_weight'] and finite(pair['pair_weight'],'pair_weight') > 0 and header['retry'] == receipt['retries'],
                'Accepted contract/weight/retry differs')
        require(header['loss_scope'] == 'actual_training_forward_before_optimizer' and header['outer_old_adapter']==receipt['outer_old_adapter'] and header['student_pre_update_adapter_sha256']==receipt['pre_update_adapter_sha256'], 'Header pre-update/old/scope differs')
        ids, mask = header['token_ids'], header['policy_mask']
        require(len(ids) == len(mask) == len(header['token_offsets']), 'header response arrays differ')
        require(all(v in (0, 1) for v in mask), 'invalid policy mask')
        positions = [i for i, value in enumerate(mask) if value]
        require(len(positions) == pair['valid_tokens'] > 0, 'policy positions differ from pair denominator')
        for line in f:
            digest.update(line); size += len(line)
            require(line.endswith(b'\n'), 'partial token line')
            row = json.loads(line)
            if 'full_vocab_dump' in row:
                require(row['full_vocab_dump'] in ('complete', 'cap_skip') and
                        type(row['token_index']) is int and row['token_index'] in positions,
                        'invalid auxiliary full-vocabulary dump row')
                stats['auxiliary_vocabulary_dump_rows'] += 1
                continue
            count = stats['valid_token_rows']
            require(count < len(positions), 'extra token metric row')
            index = positions[count]
            require(row['token_index'] == index and row['token_id'] == ids[index] and
                    row['token_offsets'] == header['token_offsets'][index] and row['policy_mask'] is True,
                    'policy token identity/order differs')
            require(row['valid_trajectory_tokens'] == len(positions) and
                    row['global_actual_pairs'] == receipt['global_actual_pairs'] and row['pair_weight'] == pair['pair_weight'],
                    'token normalization denominator differs')
            require(row['direction'] == 'student_to_fixed_target_reverse' and row['target_kind'] == ('mixture' if receipt['contract']['variant'] in ('M','J') else 'teacher')
                    and row['loss_scope'] == 'actual_training_forward_before_optimizer'
                    and row['teacher_forced_fixed_student_prefix'] is True
                    and row['actual_generation_scores'] is False, 'token loss scope differs')
            hit, vocab = row['clipped_vocabulary_count'], row['vocabulary_size']
            require(type(hit) is int and type(vocab) is int and 0 <= hit <= vocab and vocab > 0, 'invalid hit count')
            fraction = finite(row['clipped_vocabulary_fraction'], 'vocabulary fraction')
            require(math.isclose(fraction, hit / vocab, rel_tol=2e-7, abs_tol=2e-10), 'vocab fraction differs')
            stats['clipped_count'] += hit
            stats['hit_tokens'] += bool(hit)
            stats['vocabulary_slots'] += vocab
            stats['vocabulary_sizes'].add(vocab)
            values = {k: finite(row[field], field) for k, field in LOSS_KEYS.items()}
            require(values['raw'] + 2e-6 >= values['post'], 'upper clip increased scalar loss')
            for key, value in values.items():
                stats['loss_sum'][key] += value
            for key, field in MASS_KEYS.items():
                mass = finite(row[field], field)
                require(0 <= mass <= 1 + 2e-6 and (hit > 0 or mass == 0), 'invalid hit probability mass')
                stats['mass_sum'][key] += mass
                stats['mass_max'][key] = max(stats['mass_max'][key], mass)
            if receipt['contract']['variant'] not in ('M','J'):
                require(row[MASS_KEYS['teacher']] == row[MASS_KEYS['mixture']], 'Non-mixture teacher/target mass differs')
            for field in ('student_to_teacher_raw_kl', 'student_to_outer_old_raw_kl',
                          'student_to_mixture_raw_kl', 'trajectory_contribution', 'global_batch_contribution',
                          'student_observed_token_logp', 'student_entropy', 'teacher_observed_token_logp',
                          'teacher_entropy', 'outer_old_observed_token_logp', 'outer_old_entropy',
                          'mixture_observed_token_logp', 'mixture_entropy'):
                finite(row[field], field)
            require(equal_fp32_log(row['trajectory_contribution'], values['post'] / len(positions)), 'trajectory contribution differs')
            require(equal_fp32_log(row['global_batch_contribution'], pair['pair_weight'] * values['post'] / len(positions) / receipt['global_actual_pairs']),
                    'global contribution differs')
            stats['valid_token_rows'] += 1
        require(stats['valid_token_rows'] == len(positions), 'token metric rows incomplete')
        backups.finish(ref, f, s, l, path, digest.hexdigest(), size, 'committed_token_log')
    require(stats['clipped_count'] == pair['clipped_vocabulary_count'], 'pair clip count differs')
    errors = {}
    for key in LOSS_KEYS:
        mean = stats['loss_sum'][key] / len(positions)
        native = finite(pair[key + '_loss'], 'native pair ' + key)
        errors[key] = abs(mean - native)
        require(equal_fp32_log(mean, native), 'pooled token/native chunked recorded mean differs beyond stated rounding tolerance')
    require(equal_fp32_log(pair['weighted_post_loss'],pair['pair_weight']*pair['post_loss']),'Native weighted objective differs')
    stats['native_mean_rounding_errors'] = errors
    return stats


def output_stats(stats, trajectory_count, trajectory_mass_sum, trajectory_loss_sum):
    n, hit = stats['valid_token_rows'], stats['hit_tokens']
    return dict(valid_token_rows=n, clipped_vocabulary_count=stats['clipped_count'],
                auxiliary_vocabulary_dump_rows=stats['auxiliary_vocabulary_dump_rows'],
                hit_token_count=hit, hit_token_fraction=hit / n if n else None,
                vocabulary_slots=stats['vocabulary_slots'], vocabulary_sizes=sorted(stats['vocabulary_sizes']),
                pooled_vocabulary_fraction=stats['clipped_count'] / stats['vocabulary_slots'] if stats['vocabulary_slots'] else None,
                masses={key: dict(sum_over_valid_tokens=stats['mass_sum'][key],
                                  pooled_valid_token_mean=stats['mass_sum'][key] / n if n else None,
                                  hit_token_mean=stats['mass_sum'][key] / hit if hit else None,
                                  equal_trajectory_token_mean=trajectory_mass_sum[key] / trajectory_count if trajectory_count else None,
                                  maximum_token_mass=stats['mass_max'][key] if n else None) for key in MASS_KEYS},
                loss={key: dict(pooled_valid_token_mean=stats['loss_sum'][key] / n if n else None,
                                equal_trajectory_token_mean=trajectory_loss_sum[key] / trajectory_count if trajectory_count else None) for key in LOSS_KEYS},
                all_inspected_metric_fields_finite=True)


def merge_stats(destination, source):
    for key in ('valid_token_rows','clipped_count','hit_tokens','vocabulary_slots','auxiliary_vocabulary_dump_rows'):
        destination[key]+=source[key]
    destination['vocabulary_sizes'].update(source['vocabulary_sizes'])
    for key in MASS_KEYS:
        destination['mass_sum'][key]+=source['mass_sum'][key]
        destination['mass_max'][key]=max(destination['mass_max'][key],source['mass_max'][key])
    for key in LOSS_KEYS:destination['loss_sum'][key]+=source['loss_sum'][key]


def aggregate_group(spec,facts):
    variant=spec['variant'];require(variant in VARIANTS,'Unknown group')
    config,cp=stable_json(spec['config']['path'],spec['config']['sha256'])
    run=config['run_id'];require(config['variant']==variant and config['scientific']['outer_iterations']==10,'Original group config differs')
    require(cp['sha256'] in {p['sha256'] for p in facts['configs']},'Group config absent from accepted facts')
    outers=sorted([o for o in facts['outers'] if o['variant']==variant],key=lambda o:o['outer'])
    require(len(outers)==10 and [o['outer'] for o in outers]==list(range(10)) and all(o['status']=='COMPLETE_FACTS' and o['run_id']==run for o in outers),'Only a full closed ten-outer group may be analyzed')
    backups=FixedBackups(spec['backups'],run)
    jobs=[];seen=set();before=exposure=0
    for outer in outers:
        k=outer['counts']['K'];contract=outer['contract']
        require(contract==dict(direction='student_to_target_reverse',target_version='opsd-ablation-reverse-v1',variant=variant,precision=config['precision']['mode'],mixture_alpha=.1 if variant in ('M','J') else 0.),'Scientific target/precision contract differs')
        require(outer['counts']['optimizer_updates_before']==before and len(outer['updates'])==math.ceil(k/32) and outer['counts']['cumulative_pair_exposure']==exposure+k,'Outer counter chain differs')
        for n,update in enumerate(outer['updates'],1):
            require(update['update']==before+n and update['global_actual_pairs']==min(32,k-32*(n-1)),'Actual committed update/tail differs')
            key=(run,update['update']);require(key not in seen,'Duplicate committed update');seen.add(key)
            jobs.append((outer,update))
        before=outer['counts']['optimizer_updates'];exposure+=k
    receipts={}
    def read_receipt(ref):receipts[ref['path']]=backups.receipt(ref)
    backups.visit([u['source'] for o,u in jobs],read_receipt)
    requests=[];specs={};outer_states={}
    for outer in outers:
        outer_states[outer['outer']]=dict(stats=fresh_stats(),pair_count=0,pairs=set(),weights=0.,
            mass={k:0. for k in MASS_KEYS},loss={k:0. for k in LOSS_KEYS},retry_count=0)
    for outer,update in jobs:
        receipt=receipts[update['source']['path']]
        require(receipt['accepted'] is True and receipt['run_id']==run and receipt['outer_index']==outer['outer'] and receipt['optimizer_updates']==update['update'] and receipt['contract']==outer['contract'],'Native committed receipt identity differs')
        require(receipt['loss_reduction']=='trajectory_mean' and receipt['micro_batch']==1 and receipt['learning_rate']==config['scientific']['learning_rate'],'Actual loss/LR contract differs')
        require(receipt['scaler']['accepted'] is True and receipt['scaler']['adam_advanced'] is True and receipt['scaler']['overflow'] is False and type(receipt['retries']) is int and receipt['retries']>=0,'Accepted step/scaler/retry differs')
        require(receipt['global_actual_pairs']==update['global_actual_pairs'] and receipt['global_valid_tokens']==update['global_valid_tokens'],'Receipt/facts denominator differs')
        ranks=receipt['all_rank_actual_token_metrics'];require(sorted(r['rank'] for r in ranks)==list(range(8)),'Distinct eight ranks required')
        pairs=[p for r in ranks for p in r['pairs']]
        require(len(pairs)==receipt['global_actual_pairs'] and len({p['pair_index'] for p in pairs})==len(pairs) and sum(p['valid_tokens'] for p in pairs)==receipt['global_valid_tokens'],'Native actual pairs/tokens differ')
        require(sum(p['clipped_vocabulary_count'] for p in pairs)==update['metrics']['clipped_vocabulary_count'],'Native update clipcount differs')
        outer_states[outer['outer']]['retry_count']+=receipt['retries']
        for rank in ranks:
            require(rank['actual']['rank']==rank['rank'],'Actual rank identity differs')
            for pair in rank['pairs']:
                ref=pair['token_log'];require(ref['path'] not in specs,'Repeated committed token file')
                specs[ref['path']]=(outer,receipt,rank['rank'],pair);requests.append(ref)
    def read_tokens(ref):
        outer,receipt,rank,pair=specs[ref['path']];state=outer_states[outer['outer']]
        require(pair['pair_index'] not in state['pairs'],'Repeated trajectory exposure')
        state['pairs'].add(pair['pair_index'])
        result=token_log(backups,ref,pair,receipt,rank)
        state['pair_count']+=1;state['weights']+=pair['pair_weight'];merge_stats(state['stats'],result)
        for key in MASS_KEYS:state['mass'][key]+=result['mass_sum'][key]/result['valid_token_rows']
        for key in LOSS_KEYS:state['loss'][key]+=result['loss_sum'][key]/result['valid_token_rows']
    backups.visit(requests,read_tokens)
    rows=[];total=fresh_stats();tm={k:0. for k in MASS_KEYS};tl={k:0. for k in LOSS_KEYS}
    for outer in outers:
        state=outer_states[outer['outer']];k=outer['counts']['K'];st=state['stats']
        require(state['pair_count']==k and state['pairs']==set(range(k)) and st['valid_token_rows']==outer['counts']['valid_tokens'] and st['clipped_count']==outer['metrics']['clipped_vocabulary_count'],'Outer full committed cover differs')
        require(equal_fp32_log(state['weights'],outer['metrics']['weighted_pair_exposure']),'Usable weighted exposure differs')
        metrics=output_stats(st,k,state['mass'],state['loss'])
        for key in LOSS_KEYS:
            actual=metrics['loss'][key]['equal_trajectory_token_mean'];native=outer['metrics']['equal_trajectory_means'][key+'_loss']['value']
            require(actual is None and native is None or actual is not None and native is not None and equal_fp32_log(actual,native),'Outer native loss differs')
        rows.append(dict(variant=variant,outer_index=outer['outer'],status='COMPLETE',precision=config['precision']['mode'],mixture_alpha=outer['contract']['mixture_alpha'],pair_exposures=k,actual_weighted_pair_exposure=state['weights'],committed_updates=len(outer['updates']),recorded_retries=state['retry_count'],**metrics))
        merge_stats(total,st)
        for key in MASS_KEYS:tm[key]+=state['mass'][key]
        for key in LOSS_KEYS:tl[key]+=state['loss'][key]
    return dict(status='COMPLETE',variant=variant,run_id=run,actual_updates=before,pair_exposures=exposure,outers=rows,totals=output_stats(total,exposure,tm,tl)),dict(config=cp,backups=backups.inputs,files=backups.reads)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','manifest-sha256','output'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    manifest,mp=stable_json(args.manifest,args.manifest_sha256)
    require(manifest['schema']=='seven-group-closed-clipmass-inputs-v1','Explicit recorded inputs required')
    facts,fp=stable_json(manifest['facts']['path'],manifest['facts']['sha256'])
    require(facts['schema']=='opsd-ablation-outer-facts-v1' and facts['accepted'] is True,'Native accepted facts required')
    specs=manifest['groups'];require(specs and len({s['variant'] for s in specs})==len(specs) and set(s['variant'] for s in specs)<=set(VARIANTS),'Exact nonempty group subset required')
    result={};private={}
    for spec in specs:result[spec['variant']],private[spec['variant']]=aggregate_group(spec,facts)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    summary=dict(schema='seven-group-committed-clipmass-v1',accepted=True,groups=result,
        coverage=dict(completed_groups=len(result),expected_groups=7,completed_outers=10*len(result),expected_outers=70,pending_groups=[v for v in VARIANTS if v not in result],all70_token_records_complete=len(result)==7),
        definitions=dict(hit_token_fraction='Policy-valid positions with any clipped component / valid positions; not vocab fraction',pooled_vocabulary_fraction='Clipped vocabulary components / sum vocabulary size over valid positions',mass='Each independent distribution probability on same student reverse-component>10 set; exposure sums are not one probability',means='Separate pooled token, hit-token-only and equal-trajectory means; masses are never multiplied by W weights',nonmixture='mixture field is teacher target alias for groups other than M/J',loss='Native raw/post logged metrics; weighted objective separate, not gradient contribution'),
        limitations='Recorded committed forward logs only; no logits/top32/fullNPY/model/tensor/EM/causal reconstruction. Recorded retries are not all failed attempts.',
        input_digests=dict(manifest=mp['sha256'],facts=fp['sha256']))
    (out/'clipmass-publication.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    (out/'referenced-files-proof.private.json').write_text(json.dumps(dict(accepted=True,manifest=mp,facts=fp,groups=private),indent=2,allow_nan=False)+'\n')
    csvrows=[]
    for v in VARIANTS:
        if v not in result:
            csvrows.extend(dict(variant=v,outer_index=i,status='PENDING') for i in range(10));continue
        for row in result[v]['outers']:
            flat={k:row[k] for k in ('variant','outer_index','status','precision','mixture_alpha','pair_exposures','actual_weighted_pair_exposure','committed_updates','recorded_retries','valid_token_rows','clipped_vocabulary_count','hit_token_count','hit_token_fraction','pooled_vocabulary_fraction')}
            for key in MASS_KEYS:
                for field,value in row['masses'][key].items():flat[key+'_'+field]=value
            csvrows.append(flat)
    with (out/'clipmass-70-slots.csv').open('x',newline='') as f:
        fields=list(dict.fromkeys(k for r in csvrows for k in r));writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(csvrows)
    artifact={}
    for name in ('clipmass-publication.json','clipmass-70-slots.csv','referenced-files-proof.private.json'):
        raw=(out/name).read_bytes();artifact[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    code=Path(__file__).read_bytes()
    (out/'derived-manifest.json').write_text(json.dumps(dict(accepted=True,source_code_sha256=hashlib.sha256(code).hexdigest(),artifacts=artifact,coverage=summary['coverage'],no_node_or_model_IO=True),indent=2)+'\n')
    print(json.dumps(dict(accepted=True,coverage=summary['coverage'])))


if __name__=='__main__':main()
