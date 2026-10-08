"""Small real-source CPU loop fixtures; not actual final200/GPU evidence."""
import copy
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] / 'opsd-final200-paired-eval-source'
sys.path.insert(0, str(REPO))
from search_r1.baselines import final200_evaluation as final
from search_r1.baselines import corrected_evaluation as legacy
from search_r1.llm_agent import generation
from search_r1 import qa_protocol
from verl import DataProto
from verl.utils.reward_score import qa_em
from verl.utils.torch_functional import get_eos_mask

spec = importlib.util.spec_from_file_location('raw_replay_fixture_subject', HERE / 'replay_final200_raw.py')
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)


class Tokenizer:
    """Explicit synthetic tokenizer, never official-tokenizer qualification."""
    pad_token_id = 0
    eos_token_id = 1
    pad_token = '<pad>'
    def __len__(self): return 200000
    def __call__(self, text, add_special_tokens=False, return_tensors=None, padding=False):
        if isinstance(text, str): return {'input_ids':[ord(x)+2 for x in text]}
        rows = [self(t)['input_ids'] for t in text]; width = max(map(len,rows),default=0)
        if return_tensors == 'pt': return {'input_ids':torch.tensor([r+[0]*(width-len(r)) for r in rows],dtype=torch.int64)}
        return {'input_ids':rows}
    def decode(self, ids, skip_special_tokens=False):
        return ''.join(chr(int(x)-2) if int(x)>1 else ('<|im_end|>' if int(x)==1 and not skip_special_tokens else '') for x in ids)
    def batch_decode(self, rows, skip_special_tokens=False): return [self.decode(r,skip_special_tokens) for r in rows]
    def apply_chat_template(self, prompt, add_generation_prompt=True, tokenize=False):
        return ''.join('['+t['role']+']'+t['content']+'\n' for t in prompt)+'[assistant]'


TOK = Tokenizer()
META = dict(eos_token_id=1,pad_token_id=0,recompute_log_prob=False,do_sample=False,validate=True)
INVALID = c.invalid_literal((REPO/'search_r1/llm_agent/generation.py').read_bytes())


class CPUBackend:
    """Fake generation only; five output tensors follow true frozen vLLM rules."""
    def __init__(self): self.round=0; self.calls=[]
    def generate_sequences(self, probe):
        self.calls.append(dict(sequence=len(self.calls)+1,rows=len(probe),do_sample=False,validate=True))
        texts=[]
        for ids in probe.batch['input_ids']:
            q=int(re.search(r'Q([0-9]+)',TOK.decode(ids)).group(1))
            if q%4==0: text='<think>query</think><search>query</search>' if self.round==0 else '<think>answer</think><answer>A</answer>'
            elif q%4==1: text='<answer></answer>'
            elif q%4==2: text='Free reasoning <answer>A</answer>'
            else: text='invalid action'
            texts.append(text)
        self.round+=1
        responses=torch.tensor([TOK(t)['input_ids']+[1]+[0]*(499-len(TOK(t)['input_ids'])) for t in texts],dtype=torch.int64)
        prompts=probe.batch['input_ids'].clone(); positions=probe.batch['position_ids'].clone()
        return DataProto.from_dict(tensors=dict(prompts=prompts,responses=responses,
            input_ids=torch.cat([prompts,responses],dim=1),
            attention_mask=torch.cat([probe.batch['attention_mask'],get_eos_mask(responses,1)],dim=1),
            position_ids=torch.cat([positions,positions[:,-1:]+torch.arange(1,501)[None,:]],dim=1)))


def fixture(n=3,start_question=0,role='official_base'):
    prompts=[[dict(role='user',content='CPU synthetic original Q'+str(start_question+i))] for i in range(n)]
    records=[dict(ordinal=i,file_row=i,file_sha256='a'*64,question_id='cpu:'+str(i),question_uid=hashlib.sha256(str(i).encode()).hexdigest(),
        identity=dict(prompt=p,source='nq',question_prompt_text=p[0]['content'],targets=['A'])) for i,p in enumerate(prompts)]
    inputs=[TOK(TOK.apply_chat_template(p))['input_ids'] for p in prompts]; width=max(map(len,inputs))
    inputs=torch.tensor([[0]*(width-len(r))+r for r in inputs],dtype=torch.int64); att=(inputs!=0).long()
    probe=DataProto.from_dict(tensors=dict(input_ids=inputs,attention_mask=att,position_ids=(att.cumsum(1)-1)*att),meta_info=META.copy())
    backend=CPUBackend(); config=generation.GenerationConfig(max_turns=4,max_start_length=2048,max_prompt_length=4096,
        max_response_length=500,max_obs_length=500,num_gpus=8,search_url='http://127.0.0.1:18084/retrieve',topk=3,force_final_answer=False)
    manager=generation.LLMGenerationManager(TOK,backend,config,is_validation=True)
    observer=final.TurnObserver(TOK); manager.final200_observer=observer
    docs=[dict(document=dict(contents='Title\nOnly observation <answer>A</answer>'),score=1.0) for _ in range(3)]
    with mock.patch.object(generation.requests,'post',side_effect=lambda url,json:types.SimpleNamespace(json=lambda:dict(result=[copy.deepcopy(docs) for _ in json['queries']]))):
        output=manager.run_llm_loop(probe,inputs[:,-2048:].clone())
    raw=observer.consume(final.arrays_of(output.batch)); rows=[]
    for i,record in enumerate(records):
        turns=[]
        for turn in raw['turns']:
            if i not in turn['active_indices']: continue
            active_i=turn['active_indices'].index(i)
            turns.append(dict(step=turn['step'],active_indices=turn['active_indices'],
                generation_context={k:v[active_i] for k,v in turn['generation_context'].items()},
                raw_generation={k:v[active_i] for k,v in turn['raw_generation'].items()},raw_generation_meta=turn['raw_generation_meta'],
                action_token_ids=turn['action_token_ids'][i],action_text=turn['action_text'][i],observation_text=turn['observation_text'][i],
                observation_ids_untruncated=turn['observation_ids_untruncated'][i],observation_ids_retained=turn['observation_ids_retained'][i],
                done=turn['dones'][i],valid_action=turn['valid_actions'][i],search=turn['searches'][i],retrieval_step_ref=turn['step'],
                rolling_after={k:v[i] for k,v in turn['rolling_after'].items()},right_after={k:v[i] for k,v in turn['right_after'].items()}))
        rows.append(dict(kind='ACTUAL_FINAL200_QUESTION_FULL_TOOL_TRACE',model_role=role,gold_location='DRIVER_SCORING_ONLY',input_record=record,
            initial_input={k:v[i] for k,v in raw['initial_input'].items()},initial_start=raw['initial_start'][i],generation_meta=raw['generation_meta'],
            turns=turns,final={k:v[i] for k,v in raw['final'].items()},final_meta={k:v[i] if isinstance(v,list) and len(v)==n else v for k,v in raw['final_meta'].items()}))
    rows=final.canonical(rows)
    tools=final.canonical(dict(start=0,count=n,turns=[dict(step=t['step'],active_indices=t['active_indices'],retrieval=t['retrieval']) for t in raw['turns']]))
    arrays={k:v.detach().cpu().numpy().copy() for k,v in output.batch.items() if k in c.SCORING_KEYS}
    return records,rows,tools,arrays,backend.calls


def check(value,role='official_base'):
    records,rows,tools,arrays,calls=value
    expected=c.extra_raw_checks(records,rows,tools,arrays,TOK,role,INVALID)
    final.validate_raw_batch(records,rows,tools,TOK)
    return c.join_calls(expected,calls)


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.small=fixture(3); cls.last=fixture(1,3); cls.full=fixture(64)

    def test_true64_and_last1_dispatch_padding_counts(self):
        self.assertEqual(check(copy.deepcopy(self.full))['discarded_padding_output_rows'],0)
        result=check(copy.deepcopy(self.last))
        self.assertEqual(result['actual_calls'],4); self.assertEqual(result['actual_gpu_dispatch_rows'],32)
        self.assertEqual(result['real_active_generation_rows'],4); self.assertEqual(result['discarded_padding_output_rows'],28)
        self.assertIs(result['discarded_padding_outputs_replayed'],False)

    def test_active_shrink_with_padding_and_cpu_repaired_scores(self):
        value=copy.deepcopy(self.small); self.assertEqual(check(value)['discarded_padding_output_rows'],12)
        records,rows,tools,arrays,calls=value
        scores=legacy.score_arrays(arrays,records,TOK)
        self.assertEqual([r['em'] for r in scores],[1.,0.,1.])
        self.assertEqual([r['format_rate'] for r in scores],[1.,0.,0.])
        self.assertTrue(all(not row['final_meta']['active_mask'] for row in rows))
        self.assertEqual(qa_em.compute_score_generated_em(['<search>x</search>'],dict(target=['A'])),0.)

    def test_raw_pad_positions_are_not_final_cumsum_positions(self):
        value=copy.deepcopy(self.last); turn=value[1][0]['turns'][0]; raw=turn['raw_generation']
        self.assertGreater(raw['position_ids'][-1],0); check(value)
        att=np.array(raw['attention_mask']); raw['position_ids']=((att.cumsum()-1)*att).tolist()
        with self.assertRaisesRegex(ValueError,'vLLM'):check(value)

    def test_raw_five_tensor_mismatch_and_final_npz_position_rejected(self):
        for key in ('prompts','input_ids','attention_mask','position_ids'):
            value=copy.deepcopy(self.small); value[1][0]['turns'][0]['raw_generation'][key][0]+=1
            with self.assertRaises(ValueError):check(value)
        value=copy.deepcopy(self.small); value[3]['position_ids'][0,-1]+=1
        with self.assertRaisesRegex(ValueError,'NPZ'):check(value)
        value=copy.deepcopy(self.small); value[1][0]['turns'][0]['raw_generation_meta']={'gold':'A'}
        with self.assertRaisesRegex(ValueError,'output metadata'):check(value)

    def test_fixed_invalid_observation_literal_additional_join(self):
        value=copy.deepcopy(self.last); value[1][0]['turns'][0]['observation_text']='a different error message'
        with self.assertRaisesRegex(ValueError,'fixed environment literal'):check(value)

    def test_role_gold_metadata_and_bool_flags_fail_closed(self):
        for mutate in (lambda v:v[1][0].update(model_role='actual_hf200'),
            lambda v:v[1][0]['generation_meta'].update(gold='A'),
            lambda v:v[1][0]['turns'][0].update(done=False),
            lambda v:v[1][0]['final_meta'].update(valid_action_stats=True)):
            value=copy.deepcopy(self.small);mutate(value)
            with self.assertRaises(ValueError):check(value)

    def test_active_padding_phantom_and_missing_actual_call_rejected(self):
        value=copy.deepcopy(self.small);value[2]['turns'][0]['active_indices'].append(3)
        with self.assertRaises(ValueError):check(value)
        value=copy.deepcopy(self.last);value[4].pop()
        with self.assertRaisesRegex(ValueError,'generation call'):check(value)
        value=copy.deepcopy(self.last);value[4][0]['rows']=1
        with self.assertRaisesRegex(ValueError,'padded call'):check(value)

    def test_observation_only_gold_cannot_score(self):
        records,rows,tools,arrays,calls=copy.deepcopy(self.small)
        mask=arrays['attention_mask'][0,-len(arrays['responses'][0]):].astype(bool)&arrays['info_mask'][0,-len(arrays['responses'][0]):].astype(bool)
        spans=qa_protocol.decode_model_spans(TOK,arrays['responses'][0].tolist(),mask.tolist())
        self.assertFalse(any('Only observation' in s for s in spans))
        self.assertIsNone(qa_protocol.extract_final_answer(['<answer>A</answer>','<search>x</search>']))
        self.assertEqual(qa_em.compute_score_generated_em(['<answer><answer>A</answer></answer>'],dict(target=['A'])),0.)

    def test_bounded_gzip_full_sha_truncation_extra_and_partial_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder).resolve(); reader=c.Originals(os.getuid())
            raw=b'{"fixture":true}\n'; zipped=gzip.compress(raw,compresslevel=1,mtime=0)
            entry=dict(count=1,raw_bytes=len(raw),raw_sha256=hashlib.sha256(raw).hexdigest(),bytes=len(zipped),sha256=hashlib.sha256(zipped).hexdigest())
            q=p/'rows.gz';q.write_bytes(zipped)
            self.assertEqual(c.full_turn_rows(reader,str(q),entry),[dict(fixture=True)])
            for name,data in (('short.gz',zipped[:-3]),('extra.gz',zipped+gzip.compress(raw)),('partial.gz',gzip.compress(raw[:-1]))):
                q=p/name;q.write_bytes(data); bad=dict(entry,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
                with self.assertRaises(ValueError):c.full_turn_rows(reader,str(q),bad)
                self.assertEqual(q.read_bytes(),data)
            bad=dict(entry,raw_bytes=c.MAX_BATCH_RAW+1)
            with self.assertRaisesRegex(ValueError,'bound'):c.full_turn_rows(reader,str(p/'rows.gz'),bad)

    def test_original_fd_uid_sha_link_and_replacement_failures_preserve_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder).resolve();q=p/'record.json';q.write_bytes(b'{"v":1}');sha=hashlib.sha256(q.read_bytes()).hexdigest();reader=c.Originals(os.getuid())
            self.assertEqual(reader.json(dict(path=str(q),sha256=sha)),dict(v=1))
            with self.assertRaises(ValueError):reader.json(dict(path=str(q),sha256='0'*64))
            with self.assertRaises(ValueError):c.Originals(os.getuid()+1).json(dict(path=str(q),sha256=sha))
            link=p/'link';link.symlink_to(q)
            with self.assertRaises(ValueError):reader.json(dict(path=str(link),sha256=sha))
            alias=p/'hardlink';os.link(q,alias)
            with self.assertRaises(ValueError):reader.json(dict(path=str(q),sha256=sha))
            alias.unlink()
            with self.assertRaises(ValueError):
                with reader.opened(str(q),sha):
                    replacement=p/'other';replacement.write_bytes(b'{"v":1}');os.replace(replacement,q)
            self.assertEqual(q.read_bytes(),b'{"v":1}')

    def test_npz_object_or_wrong_member_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder).resolve()/'rows.npz';arrays=self.small[3]
            with p.open('wb') as f:np.savez_compressed(f,**arrays)
            reader=c.Originals(os.getuid());sha=hashlib.sha256(p.read_bytes()).hexdigest()
            self.assertEqual(set(c.npz_arrays(reader,str(p),sha)),c.SCORING_KEYS)
            for altered in (dict(arrays,unexpected=np.array([1])),dict(arrays,responses=np.array([[object()]],dtype=object))):
                with p.open('wb') as f:np.savez_compressed(f,**altered)
                sha=hashlib.sha256(p.read_bytes()).hexdigest()
                with self.assertRaises(ValueError):c.npz_arrays(c.Originals(os.getuid()),str(p),sha)

    def test_json_duplicate_nonfinite_and_atomic_no_overwrite(self):
        for raw in (b'{"v":1,"v":2}',b'{"v":NaN}'):
            with self.assertRaises(ValueError):c.loads(raw)
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder).resolve()/'out.json';c.write_once(p,b'original')
            with self.assertRaises(FileExistsError):c.write_once(p,b'changed')
            self.assertEqual(p.read_bytes(),b'original')

    def test_source_only_empty_package_and_nonempty_raw_boundary(self):
        with tempfile.TemporaryDirectory() as folder:
            repo=Path(folder).resolve(); (repo/'search_r1').mkdir(); path=repo/'search_r1/__init__.py';path.write_bytes(b'')
            record={'sha256':hashlib.sha256(b'').hexdigest(),'bytes':0};reader=c.Originals(os.getuid())
            importer=c.PinnedRepoImporter(repo,{'search_r1/__init__.py':record},reader)
            module=types.ModuleType('search_r1');module.__spec__=types.SimpleNamespace(origin=str(path))
            importer.exec_module(module);reader.recheck()
            self.assertEqual(importer.loaded['search_r1']['sha256'],record['sha256'])
            with self.assertRaises(ValueError):reader.json({'path':str(path),'sha256':record['sha256']})
            # Native atomic JSON shape has no LF. Whole-file JSON parsing accepts it.
            path=repo/'native.json';path.write_bytes(final.encoded({'version':1,'count':1}))
            self.assertFalse(path.read_bytes().endswith(b'\n'))
            self.assertEqual(reader.json({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}),{'version':1,'count':1})


class DeclaredRuntimeBoundaryTests(unittest.TestCase):
    """Declared evidence fixtures exercise added checks, not actual worker authenticity."""
    @staticmethod
    def header():
        fixture_spec=importlib.util.spec_from_file_location('sealed_contract_fixture',HERE.parent/'final200-paired-launch-candidate/paired_eval_contract.py')
        contract=importlib.util.module_from_spec(fixture_spec);fixture_spec.loader.exec_module(contract)
        budget=copy.deepcopy(contract.BUDGET);input_ref={'path':'/cpu-fixture/input.json','sha256':'a'*64}
        stage={'kind':'REVIEWED_FINAL200_PAIRED_EVAL_STAGE','model_role':'official_base','training_commit':c.TRAIN,
            'training_job_id':'3304','actual_outer_updates':200,'eval_commit':c.EVAL,'common_manifest_sha256':c.COMMON,
            'official_base_revision':c.OFFICIAL_REVISION,'official_base_manifest_sha256':c.OFFICIAL_MANIFEST,
            'model_directory':c.BASE,'input_manifest_sha256':input_ref['sha256'],'generation_budget':budget,
            'generation_budget_sha256':hashlib.sha256(c.encoded(budget)).hexdigest()}
        manifest={'version':1,'scope':'FULL_FINAL200_TOOL_TURNS_AND_NATIVE_SCORING_VIEW','count':51713,
            'source_counts':copy.deepcopy(c.COUNTS),'input_manifest_sha256':input_ref['sha256'],'provenance':copy.deepcopy(stage),
            'lifecycle_scope':'EVALUATION_OUTPUT_COMPLETE_NOT_SCHEDULER_TERMINAL_OR_RESOURCE_RELEASE'}
        return manifest,stage,input_ref

    def test_fixed_full_role_header_budget_and_lifetime_scope(self):
        m,s,i=self.header();self.assertEqual(c.check_header(m,s,'official_base',i)['generation_budget'],s['generation_budget'])
        for mutate in (lambda m,s:m.update(count=True),lambda m,s:m.update(count=51712),
            lambda m,s:m['source_counts'].update(nq=3611),lambda m,s:m.update(lifecycle_scope='RESOURCE_RELEASED'),
            lambda m,s:s.update(model_directory=c.HF200),lambda m,s:s['generation_budget'].update(max_turns=5),
            lambda m,s:s['generation_budget'].update(**{'retriever.url':'http://remote-node/retrieve'})):
            m,s,i=self.header();mutate(m,s)
            # Keep declaration copies and budget digest joined so the targeted contract check, not stale copy, rejects it.
            s['generation_budget_sha256']=hashlib.sha256(c.encoded(s['generation_budget'])).hexdigest();m['provenance']=copy.deepcopy(s)
            with self.assertRaises(ValueError):c.check_header(m,s,'official_base',i)
        role=c.RoleReplay.__new__(c.RoleReplay);role.manifest={'files':[{}]*808,'full_turn_files':[{}]*808}
        with self.assertRaisesRegex(ValueError,'809'):next(role.walk(TOK))

    @staticmethod
    def runtime():
        # Fresh synthetic allocation ID intentionally has no claim to a real scheduler job.
        allocation={'job_id':'987654321','step_id':'0','node':'h100-10','uid':0,'gpu_uuids':['GPU-CPU-FIXTURE-'+str(i) for i in range(8)]}
        prov={'eval_job_id':allocation['job_id'],'source_files':{'search_r1/baselines/corrected_evaluation.py':'a'*64}}
        rows=[];starts=[];finishes=[]
        for rank in range(8):
            process={'pid':10000+rank,'process_start_ticks':str(20000+rank),'uid':0,'uids':[0]*4,
                'cgroup':'0::/slurm/job_987654321/step_0/task_'+str(rank)}
            rows.append({'rank':rank,'world_size':8,'pid':process['pid'],'uid':0,'process':process,
                         'gpu_uuid':allocation['gpu_uuids'][rank]})
            clock={'state_entries':0,'steps':[],'scheduler':{'last_epoch':0}}
            report={'rank':rank,'world_size':8,'process':copy.deepcopy(process),'before':copy.deepcopy(clock),'after':copy.deepcopy(clock),
                'attempted_optimizer_steps':0,'attempted_scheduler_steps':0,'source_sha256':c.FINAL_MODULE_SHA,
                'scope':'INDEPENDENT_EVAL_WORKER_ZERO_STEPS_NOT_TRAINING_ADAM_RESTORE'}
            starts.append(dict(copy.deepcopy(report),phase='install'));finishes.append(dict(copy.deepcopy(report),phase='finish'))
        calls=[{'sequence':1,'rows':8,'do_sample':False,'validate':True}]
        audit={'worker_audit':{'spec':{'job_id':allocation['job_id'],'eval_repo':c.REPO,'uid':0},'rows':rows},'generation_calls':calls,'rejected_calls':0}
        manifest={'provenance':prov,'zero_update_guard':{'attempted_update_dispatches':0,
            'guarded_dispatches':['actor_rollout_wg.update_actor'],'original_dispatches':[{'fixture':True}],
            'guard_source_sha256':'a'*64,'corrected_runtime':audit},'actual_eval_worker_zero_optimizer':{'start':starts,'finish':finishes}}
        helpers={'runtime':types.SimpleNamespace(validate_completed_runtime=lambda audit,prov:None)}
        return manifest,allocation,helpers

    def test_added_typed_zero_clock_pid_cgroup_and_uuid_boundaries(self):
        m,a,h=self.runtime();self.assertEqual(c.validate_workers(m,a,h),m['zero_update_guard']['corrected_runtime']['generation_calls'])
        edits=(lambda m,a:m['zero_update_guard']['corrected_runtime'].update(rejected_calls=False),
            lambda m,a:m['actual_eval_worker_zero_optimizer']['finish'][0].update(attempted_optimizer_steps=True),
            lambda m,a:m['actual_eval_worker_zero_optimizer']['finish'][0]['after']['scheduler'].update(last_epoch=1),
            lambda m,a:m['zero_update_guard']['corrected_runtime']['worker_audit']['rows'][0].update(rank=False),
            lambda m,a:m['zero_update_guard']['corrected_runtime']['worker_audit']['rows'][0]['process'].update(cgroup='0::/slurm/job_987654321/step_00/task_0'),
            lambda m,a:m['zero_update_guard']['corrected_runtime']['worker_audit']['rows'][0].update(gpu_uuid='GPU-FOREIGN'),
            lambda m,a:a.update(job_id='3304'))
        for edit in edits:
            m,a,h=self.runtime();edit(m,a)
            with self.assertRaises(ValueError):c.validate_workers(m,a,h)


if __name__=='__main__':
    unittest.main(verbosity=2)
