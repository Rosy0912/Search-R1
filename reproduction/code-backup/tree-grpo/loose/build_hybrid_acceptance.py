import pathlib,json,hashlib
root='/dataset/shaoshitong/search-r1-rl/tree-grpo'
p=pathlib.Path('work/aux-gate-review/HYBRID_REAL_RETRIEVAL_REVIEW.md')
assert p.exists(),'Independent real result review required'
h=json.loads(pathlib.Path('outputs/evidence/hybrid-http-difference-2876.json').read_text());b=json.loads(pathlib.Path('outputs/evidence/hybrid-binary-index-difference.json').read_text())
roles={
 'binary_report_file':(root+'/logs/binary-index-difference.json','outputs/evidence/hybrid-binary-index-difference.json'),
 'hybrid_http_report_file':(root+'/runs/hybrid-cpu-gate-2876/http-difference.json','outputs/evidence/hybrid-http-difference-2876.json'),
 'index_source_file':(root+'/tools/index_gpu_server.py','work/index_gpu_server.py'),
 'hybrid_source_file':(root+'/tools/hybrid_retrieval_server.py','work/hybrid_retrieval_server.py'),
 'binary_probe_source_file':(root+'/tools/probe_binary_index.py','work/probe_binary_index.py'),
 'http_probe_source_file':(root+'/tools/probe_aux_http.py','work/probe_aux_http.py'),
 'baseline_file':(root+'/runs/retrieval-2854/cpu.json','outputs/evidence/retrieval-cpu-2854.json'),
 'independent_review_file':(root+'/logs/hybrid-actual-result-independent-review.md',str(p)),
 'stage_binding_review_file':(root+'/logs/hybrid-stage-binding-independent-review.md','work/aux-gate-review/HYBRID_STAGE_GATE_REVIEW.md'),
 'stage_checker_source_file':(root+'/tools/check_hybrid_retriever.py','work/check_hybrid_retriever.py'),
 'stage_launcher_source_file':(root+'/tools/train_tree_stage.sh','work/train_tree_stage.sh'),
 'dispatcher_source_file':(root+'/tools/dispatch_two_hybrid.sh','work/dispatch_two_hybrid.sh'),
 'cpu_sampler_source_file':(root+'/tools/sample_retriever_cpu.py','work/sample_retriever_cpu.py'),
}
r={'status':'ACCEPTED_AFTER_HYBRID_BINARY_HTTP_DIFFERENCE_REVIEW','url':'http://127.0.0.1:18082/retrieve','index_url':'http://127.0.0.1:18083','index_instance':b['instance'],'provenance':h['provenance'],'scope':'Limited fixed14 query top3 identity/content and repeated512 batching sanity; permits next two-update smoke only, not5/500/effect acceptance','retired_cpu_gate_job':2876,'retired_cpu_gate_exit_code':0,'true_top3_exact_14':14,'true_top3_exact_512':512,'top50_id_rank_exact_14':11,'top50_content_rank_exact_14':14,'replicated_512_contains_unique_queries':14,'reviewed_numeric_delta_max_top3':8.940696716308594e-7,'evidence_file_sha256':{}}
for role,(remote,local) in roles.items():r[role]=remote;r['evidence_file_sha256'][remote]=hashlib.sha256(pathlib.Path(local).read_bytes()).hexdigest()
p=pathlib.Path('outputs/evidence/hybrid-retrieval-acceptance.json');assert not p.exists();p.write_text(json.dumps(r,indent=2)+'\n');sha=hashlib.sha256(p.read_bytes()).hexdigest();pathlib.Path('work/hybrid-retrieval-acceptance.sha256').write_text(sha+'\n');print(sha)
