import copy
import unittest
from unittest.mock import patch
from . import official_treegrpo_final as f


def fixture():
    hf=dict(files={'model.safetensors':dict(sha256='a'*64)})
    plan=dict(protocol='official_treegrpo_final51713_pj_v1',group='Tree-GRPO',outer=179,
              repeat_count=3,repeat_index=1,base=f.BASE,backend='pj_pt',model=f.MODEL,
              training_uid=f.TRAIN_UID,checkpoint_identity=f.checkpoint_identity(),job=f.job_name(1),
              output=str(f.output_root(1)/'output'),evidence={n:dict(path=f.INPUTS+'/'+n,sha256=h) for n,h in f.EVIDENCE.items()},
              model_files={n:r['sha256'] for n,r in hf['files'].items()},data_root=f.BASE+'/common-data',
              data_manifest_sha256=f.DATA_SHA,ordered_ids_sha256=f.IDS_SHA,final_content_identity='b'*64,
              canonical_tokenizer=f.BASE+'/assets/model',inference_model=f.BASE+'/official-treegrpo/final-seven-model-view-v1',
              canonical_tokenizer_files={n:f.BASE_FILES[n] for n in ['tokenizer.json','tokenizer_config.json','vocab.json','merges.txt']},
              retriever_receipt=f.BASE+'/receipts/pj-retriever-ready.json',retriever_receipt_sha256='c'*64,
              allocation=copy.deepcopy(f.ALLOCATION),no_auto_retry=True,evaluation_code_commit='d'*40,evaluation_run_id='e'*32)
    return plan,hf


class FinalContracts(unittest.TestCase):
    def test_real_three_names_and_pj_limit(self):
        names=[f.job_name(i) for i in (1,2,3)]
        self.assertEqual(len(set(names)),3)
        self.assertTrue(all(len(n)<=32 for n in names))
        for i in (0,4,True,1.0):
            with self.assertRaises(ValueError):f.job_name(i)

    def test_valid_fixed_each_repeat(self):
        for i in (1,2,3):
            p,h=fixture();p.update(repeat_index=i,job=f.job_name(i),output=str(f.output_root(i)/'output'))
            self.assertEqual(f.validate_plan(p,h),f.output_root(i))

    def test_no_fake_own_method_milestone(self):
        p,h=fixture();p.update(group='original',outer=200)
        with self.assertRaises(ValueError):f.validate_plan(p,h)

    def test_no_float_or_directory_label_as_update_count(self):
        for outer in (179.0,180,200):
            p,h=fixture();p['outer']=outer
            with self.assertRaises(ValueError):f.validate_plan(p,h)

    def test_no_fourth_repeat_or_changed_source(self):
        for key,value in [('repeat_index',4),('repeat_count',5),('evaluation_code_commit','unknown'),('evaluation_run_id','bad')]:
            p,h=fixture();p[key]=value
            with self.assertRaises(ValueError):f.validate_plan(p,h)

    def test_unknown_science_and_affinity_fields_rejected(self):
        for key in ('force_final_answer','include','communication_transport'):
            p,h=fixture();p[key]='changed'
            with self.assertRaises(ValueError):f.validate_plan(p,h)

    def test_resource_changes_rejected(self):
        for key,value in [('gpus',4),('cpus',60),('memory_gib',512),('gpu_memory_gib',48)]:
            p,h=fixture();p['allocation'][key]=value
            with self.assertRaises(ValueError):f.validate_plan(p,h)

    def test_hf_paths_and_identity_rejected(self):
        for key,value in [('model','/another/model'),('training_uid','wrong'),('checkpoint_identity','0'*64),
                          ('output','/wrong/output'),('job','other'),('data_root','/other'),('backend','slurm_lumia')]:
            p,h=fixture();p[key]=value
            with self.assertRaises(ValueError):f.validate_plan(p,h)

    def test_evidence_and_tokenizer_change_rejected(self):
        p,h=fixture();p['evidence']['native-acceptance.json']['sha256']='0'*64
        with self.assertRaises(ValueError):f.validate_plan(p,h)
        p,h=fixture();p['canonical_tokenizer_files']['tokenizer.json']='0'*64
        with self.assertRaises(ValueError):f.validate_plan(p,h)

    def test_gate_binds_uid_source_plan_and_allocation(self):
        p,h=fixture();sub=dict(job=p['job'],uid='actual-uid',plan_sha256='a'*64)
        gate=dict(plan_sha256=sub['plan_sha256'],source=p['evaluation_code_commit'],actual=dict(name=p['job'],uid='actual-uid',state='RUNNING',resources=copy.deepcopy(f.ALLOCATION)))
        self.assertEqual(f.validate_gate(p,sub,gate)['uid'],'actual-uid')
        for key,value in [('uid','other'),('state','PENDING'),('name','different')]:
            bad=copy.deepcopy(gate);bad['actual'][key]=value
            with self.assertRaises(ValueError):f.validate_gate(p,sub,bad)
        for key,value in [('source','0'*40),('plan_sha256','0'*64)]:
            bad=copy.deepcopy(gate);bad[key]=value
            with self.assertRaises(ValueError):f.validate_gate(p,sub,bad)

    def test_native_hf_reload_lineage(self):
        native=dict(uid=f.TRAIN_UID,source=f.TRAIN_SOURCE,actual_iterations=179,directory_label=180,
                    actual_adam_per_rank=1432,status='CPU_ACTOR_NATIVE_FILE_TENSOR_ADAM_RNG_SCHEDULER_AUDIT_PASSED',
                    local_small_backup_verified=True,files={str(i):{} for i in range(24)},
                    workers=[dict(adam_min=1432,adam_max=1432,scheduler=dict(last_epoch=179)) for _ in range(8)])
        hf=dict(uid=f.TRAIN_UID,status='FINAL_HF_OFFICIAL_CPU_MERGE_ALL_TENSORS_AND_METADATA_ACCEPTED',
                local_small_backup_verified=True,target=f.MODEL,hf_file_identity_sha256=f.HF_ID,native_acceptance_sha256=f.EVIDENCE['native-acceptance.json'],
                parameter_comparison=dict(exact_native_bf16_projection_all_parameters=True,logical_parameters=435,rank_parameter_comparisons=3480,parameters_different_from_initialization=['test']))
        reload=dict(uid=f.TRAIN_UID,status='ACTUAL_HF_CPU_RELOAD_ALL435_PARAMETERS_AND_TIED_STORAGE_VERIFIED',hf_file_identity_sha256=f.HF_ID,logical_parameters_compared=435,physical_tied_tensors_equal=True,loaded_tied_parameter_pointer_shared=True,export_acceptance_sha256=f.EVIDENCE['hf-acceptance.json'])
        export=dict(status='EXACT_BASE_PROTOCOL_INPUTS_VERIFIED',questions=f.TOTAL,config={},config_digest=f.digest({}))
        with patch.object(f,'CONFIG_DIGEST',f.digest({})):
            f.validate_lineage(native,hf,reload,export)
            for index,key,value in [(0,'uid','wrong'),(0,'actual_iterations',180),(1,'native_acceptance_sha256','bad'),(2,'physical_tied_tensors_equal',False),(3,'questions',512),(3,'config',dict(changed=True))]:
                data=copy.deepcopy([native,hf,reload,export]);data[index][key]=value
                with self.assertRaises(ValueError):f.validate_lineage(*data)


if __name__=='__main__':unittest.main()
