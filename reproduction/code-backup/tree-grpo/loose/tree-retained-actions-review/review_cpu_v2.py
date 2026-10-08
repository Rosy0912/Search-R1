import pathlib,sys,json,copy,hashlib,contextlib,io
D=pathlib.Path(__file__).resolve().parent;W=D.parent;sys.path.insert(0,str(W))
import analyze_tree_retained_actions as m
import test_analyze_tree_retained_actions as t
source=W/'analyze_tree_retained_actions.py';src=source.read_bytes();compile(src,str(source),'exec');compile((W/'test_analyze_tree_retained_actions.py').read_bytes(),'owner_tests','exec')
with contextlib.redirect_stdout(io.StringIO()) as buf:t.main()
base=json.loads(buf.getvalue());assert base['count']==10
checks=[]
def reject(label,row):
 try:t.result(row)
 except (AssertionError,KeyError,TypeError):checks.append({'case':label,'rejected':True});return
 raise AssertionError(label+' accepted')
q=t.leaf([t.action([],[],[10]),t.action([1],[1],[10])]);a=t.result(q);assert a['own_record_exact_token_coverage']==1;checks.append({'case':'empty_action_then_valid_action_no_minus_zero_slice','passed':True})
reject('all_actions_empty',t.leaf([t.action([],[],[10])]))
q=t.leaf([t.action([1],[1],[10])]);q['metadata']['action_records'][0].pop('raw_generation_token_ids');reject('missing_raw',q)
q=t.leaf([t.action([1],[1],[10])]);q['metadata']['action_records'][0]['scoring_context_matches_generation']=False;reject('false_flag_with_equal_context',q)
q=t.leaf([t.action([1],[1],[10])]);q['metadata']['action_records'][0]['observation_tokens']=-1;reject('negative_count',q)
q=t.leaf([t.action([2**62],[2**62],[10])]);a=t.result(q);assert a['own_record_exact_token_coverage']==1;checks.append({'case':'large_int_token_ids_exact_record_only_not_vocab_claim','passed':True})
q=t.leaf([t.action([1],[1],[10])]);q['metadata']['action_records'][0]['raw_generation_token_ids']=[-1];reject('negative_raw_token',q)
q=t.leaf([t.action([1],[1],[10])]);q['row']=False;q['metadata']['action_records'][0]['action_tokens']=True
reject('bool_row_and_action_count',q)
q=t.leaf([t.action([1,2,3],[1,2],[10])]);assert t.result(q)['counts']['first_difference_is_final_action_token_records']==0
checks.append({'case':'raw_exhaustion_not_token_value_mismatch','passed':True})
proof={'status':'SOURCE_CPU_REVIEW_FIXED_PASS','source_sha256':hashlib.sha256(src).hexdigest(),'owner_ten_tests':base,'additional_checks':checks,'resolved_findings':['bool row/count rejected','raw exhaustion separated from actual final-token value mismatch'],'assets_loaded':False,'torch_imported': 'torch' in sys.modules,'gpu_calls':0,'scope':'Source-only read-only CPU fixtures, no actual assets, no final classification acceptance'}
(D/'SOURCE_CPU_V2_PROOF.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
