import importlib.util,json,pathlib,numpy as np
s=importlib.util.spec_from_file_location('reader',pathlib.Path(__file__).with_name('review.py'));r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
def fixture(d):
 t=dict(nq=300,questions=[str(i) for i in range(300)],gens=['<answer>x</answer>']+['noanswer']*1499,flat_qi=[i for i in range(300) for _ in range(5)],valid=[0],cb_ans=['x']*300,q_cb=[0]*300,commit_ctx=['c'],ans_ctx=['a'],prompts=['p']*300,flip_ctx=['f'],flip2_ctx=['ff'],answers=['x'],ems=[1],flip_ans=['y'],flips=[1],flips2=[1])
 (d/'generation-trace.json').write_text(json.dumps(t));return t

def test_missing_B_remains_incomplete(tmp_path):
 fixture(tmp_path);out=r.inspect(tmp_path)
 assert out['status']=='INCOMPLETE' and 'phase-B-receipt.json' in out['missing'] and out['checks']['generation_structure']['filtered']==1499

def test_duplicate_questions_is_real_contradiction(tmp_path):
 t=fixture(tmp_path);t['questions'][1]=t['questions'][0];(tmp_path/'generation-trace.json').write_text(json.dumps(t))
 assert any('unique' in x['error'] for x in r.inspect(tmp_path)['errors'])

def test_wrong_valid_selection_refused(tmp_path):
 t=fixture(tmp_path);t['valid']=[1];(tmp_path/'generation-trace.json').write_text(json.dumps(t))
 assert any('filtering' in x['error'] for x in r.inspect(tmp_path)['errors'])

def test_mask_nonleft_padding_refused(tmp_path):
 fixture(tmp_path);(tmp_path/'hidden-input-token-trace.json').write_text(json.dumps({k:[{'input_ids':[[1,2,3]],'attention_mask':[[1,0,1]]}] for k in ('commit_ctx','ans_ctx','prompts')}))
 assert any('left-padding' in x['error'] for x in r.inspect(tmp_path)['errors'])

def test_npz_true_label_mismatch_not_hidden_by_finite_shapes(tmp_path):
 fixture(tmp_path);(tmp_path/'frozen-input-config.json').write_text(json.dumps({'dump_args':{'tag':'nq','offset':0},'uuids':list(range(8))}));(tmp_path/'dump').mkdir()
 a={'y_em':np.array([0],np.int8),'y_flip':np.array([1],np.int8),'y_flip2':np.array([1],np.int8),'q_cb':np.array([0],np.int8),'gid':np.array([0],np.int32),'ds':np.array(['nq'])}
 a.update({f'{p}_{l}':np.ones((1,5120),np.float16) for p in ('Xc','Xa','Xp') for l in (21,24,27)});np.savez(tmp_path/'dump/nq_0.npz',**a)
 assert any('label/gid mismatch y_em' in x['error'] for x in r.inspect(tmp_path)['errors'])
