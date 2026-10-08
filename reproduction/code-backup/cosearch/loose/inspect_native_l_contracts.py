import json,collections,pathlib,hashlib
r=pathlib.Path("/dataset/shaoshitong/search-r1-rl/cosearch/author-native/runs/author-native-warm-five-l/evidence")
acks=[];contracts=[];results=[];controller=[]
for p in (r/"workers").glob("*.jsonl"):
 for line in p.read_text().splitlines():
  x=json.loads(line)
  if x.get("kind")=="ack":acks.append(x)
  elif x.get("kind")=="ranker_inference_contract":contracts.append(x)
  elif x.get("kind")=="ranker_ranking_result":results.append(x)
for p in r.glob("controller-*.jsonl"):
 for line in p.read_text().splitlines():
  x=json.loads(line)
  if x.get("kind") not in ["global_pre_update"]:controller.append(x)
groups=collections.defaultdict(list)
for x in results:groups[(x["query"],tuple(x["actual_candidate_ids"]))].append(tuple(x["actual_reranked_docs"]))
a=dict(acks=acks,contracts=contracts,ranking_results=results,controller_events=controller)
p=r/"raw-diagnosis/ranker-contract-evidence.json";p.write_text(json.dumps(a))
summary=dict(ack_count=len(acks),roles_ranks=[(x["role"],x["rank"]) for x in acks],distinct_UUIDs=len(set(x["uuid"] for x in acks)),fresh_Adam_initial=[x["adam_state_initial"] for x in acks],contracts=len(contracts),temperatures=dict(collections.Counter(str(x["actual_sampling_params"]) for x in contracts)),ranking_results=len(results),errors=dict(collections.Counter(str(x.get("original_error_type")) for x in results)),query_candidate_groups=len(groups),repeated_same_input_groups=sum(len(v)>1 for v in groups.values()),distinct_ranking_same_input_groups=sum(len(set(v))>1 for v in groups.values()),context_refusals=len(list((r/"context-refusals").glob("*"))),evidence_bytes=p.stat().st_size,evidence_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),controller_events=controller)
q=p.with_name("ranker-contract-summary.json");q.write_text(json.dumps(summary));print(json.dumps(summary))
