import json,pathlib,sys
import pytest
sys.path.insert(0,str(pathlib.Path(__file__).parent))
import prepare_run as p

def initial(tmp):
 manifest=tmp/'files.json';manifest.write_text('{}')
 return dict(retriever_command=['python','-c','originalnativewrapper','oldinitialserver','--device','cuda','--faiss_gpu'],experiment='initial',base_model='sameofficialbase',training_command=['python','-m','verl.trainer.main_ppo_eco','trainer.experiment_name=initial','trainer.default_local_dir=original','trainer.total_training_steps=101','trainer.save_freq=25','trainer.g1_refresh_freq=25','data.train_batch_size=128','actor_rollout_ref.actor.ppo_micro_batch_size=64'],source_bindings=[dict(kind='pipeline')],refit_args={'data':'same300','nq':300},probe_command=['python','probe.py','--manifest','old'])
@pytest.mark.parametrize('mode,steps,save',[('profile2',3,2),('train100',101,25)])
def test_actual_derive_preserves_math_and_samebase_not_profile_resume(tmp_path,monkeypatch,mode,steps,save):
 monkeypatch.setattr(p,'ROOT',tmp_path)
 original=initial(tmp_path)
 # Real helper reads actual bound manifest; create target pipeline under own root.
 pipeline=tmp_path/'pipeline';pipeline.mkdir();(pipeline/'files.json').write_text('{}')
 out=p.derive(original,mode,tmp_path/'new',pipeline,{'path':'actualfrozen'})
 assert out['base_model']==original['base_model'] and out['initial_calibration_reuse']=={'path':'actualfrozen'}
 assert 'trainer.total_training_steps='+str(steps) in out['training_command']
 assert 'trainer.save_freq='+str(save) in out['training_command']
 assert all(t in out['training_command'] for t in ['trainer.g1_refresh_freq=25','data.train_batch_size=128','actor_rollout_ref.actor.ppo_micro_batch_size=64'])
 assert out['retriever_command'][3]==str(pipeline/'retrieval_server.py') and out['retriever_command'][:3]==original['retriever_command'][:3] and out['retriever_command'][4:]==original['retriever_command'][4:]
 assert out['independent_gate']['sha256'] is None and original['experiment']=='initial'

def test_actual_derivation_refuses_missing_original_recipe_override(tmp_path,monkeypatch):
 monkeypatch.setattr(p,'ROOT',tmp_path);cfg=initial(tmp_path);cfg['training_command']=cfg['training_command'][:-4]
 d=tmp_path/'p';d.mkdir();(d/'files.json').write_text('{}')
 with pytest.raises(ValueError,match='missing original'):p.derive(cfg,'profile2',tmp_path/'o',d,{'path':'f'})


@pytest.mark.parametrize('mode',['profile2','train100'])
def test_original_derive_retains_new_stage_scoped_producer_and_actual_round0_refs(tmp_path,monkeypatch,mode):
 monkeypatch.setattr(p,'ROOT',tmp_path);original=initial(tmp_path)
 original.update(producer='own/scoped-producer-v4',producer_manifest_sha256='actual-source-cpu-pinned')
 original['source_bindings'].append({'kind':'producer','root':original['producer'],'manifest':{'path':'own/scoped-producer-v4/files.json','sha256':original['producer_manifest_sha256']}})
 pipeline=tmp_path/'pipeline';pipeline.mkdir();(pipeline/'files.json').write_text('{}')
 ref={'path':'ACTUAL_INITIAL_PENDING_IN_CPU_FIXTURE','sha256':'CPU_FIXTURE_NOT_REAL_HEADS'}
 out=p.derive(original,mode,tmp_path/'new',pipeline,ref)
 assert out['producer']==original['producer'] and out['producer_manifest_sha256']==original['producer_manifest_sha256']
 assert next(x for x in out['source_bindings'] if x['kind']=='producer')==original['source_bindings'][-1]
 assert out['initial_calibration_reuse']==ref and out['base_model']==original['base_model']
