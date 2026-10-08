import importlib.util,pathlib,json,pytest
s=importlib.util.spec_from_file_location('aggregate',pathlib.Path(__file__).with_name('aggregate.py'));a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
def receipts(p):
 rows=[]
 for i in range(2):
  r={'status':'CALL_REPLAY_MATCH','all_calls_verified':False,'selected_call_indices':[i],'total_original_call_count':2,'total_original_output_rows':4,'reader_sha256':'source','raw_refs':{'raw':'same'},'source_refs':{'original':'same'},'tokenizer_files':{'tokenizer':'same'},'labels':{'original':'same'},'native_defaults':{},'native_versions':{},'original_cache_source_sha256':'cache','mismatch_count':0,'prompt_mismatch_count':0,'cuda_initialized':False,'vllm_imported':False,'generated_output_rows':2,'output_tokens':3,'missing_B_not_accepted':True};q=p/str(i);q.write_text(json.dumps(r));rows.append(q)
 return rows
def test_complete_singlecalls_does_not_approve_calibration(tmp_path):
 r=a.combine(receipts(tmp_path));assert r['rows']==4 and r['missing_B_not_accepted'] and 'NOT_CALIBRATION_APPROVAL' in r['scope']
@pytest.mark.parametrize('violation',['missing','duplicate','mixed_source','mismatch'])
def test_missing_duplicate_or_changed_receipt_cannot_form_complete_match(tmp_path,violation):
 rows=receipts(tmp_path)
 if violation=='missing':rows=rows[:1]
 elif violation=='duplicate':rows=[rows[0],rows[0]]
 else:
  r=json.loads(rows[1].read_text());r['reader_sha256']='changed' if violation=='mixed_source' else r['reader_sha256'];r['mismatch_count']=1 if violation=='mismatch' else 0;rows[1].write_text(json.dumps(r))
 with pytest.raises(ValueError):a.combine(rows)
