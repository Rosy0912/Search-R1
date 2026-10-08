from pathlib import Path
import json
import time

base = Path(__file__).parent
k1 = json.loads((base / 'hkust1/identity.json').read_text())
k4 = json.loads((base / 'hkust4/identity.json').read_text())
k4_index_path = base / 'index-k4-fresh.json'
k4_index = json.loads(k4_index_path.read_text()) if k4_index_path.exists() else None
index_sha = '69c98463fdb41fc08737d88513c597725f311c44f7ba6dca4b05d8c7c658d166'
corpus_sha = '85a787a692e73fdd657e86411bfd1ac810ed193b1eea4a478de449d4871c06b9'
scanner_sha = '9b5e50257261ce16d363e58900465ad154d6158f014b7c9707e9880f0a2e84ca'
comparisons = [
    {'kind': 'index',
     'path': '/data/shaoshitong/search-e1-retriever-gpu-20261004/assets/index/e5_Flat.index',
     'k4_path': '/weka_3/shaoshitong/search-e1-retriever-stream-20261004/assets/index/e5_Flat.index',
     'bytes': 64559075373, 'k1_sha256': index_sha,
     'k4_sha256': k4_index['sha256'] if k4_index else None,
     'k1_measurement_time': 1791354116.877514,
     'k4_measurement_time': k4_index['complete_time'] if k4_index else None,
     'k1_evidence': 'Fresh streaming SHA256, full file, tool output in this subagent turn; 58.176883697509766 seconds, no GPU.',
     'k4_evidence': 'Fresh streaming SHA256 from index-k4-fresh.json.' if k4_index else 'Pending fresh complete-file hash; known manifest hash is deliberately not substituted.',
     'k4_full_hash_receipt': '/weka_3/shaoshitong/search-e1-retriever-analysis-20261007/cpu-encoder-identity/index-k4-fresh.json'},
    {'kind': 'corpus',
     'path': '/data/shaoshitong/search-e1-retriever-gpu-20261004/assets/corpus/wiki-18.jsonl',
     'k4_path': '/weka_3/shaoshitong/search-e1-retriever-stream-batched-20261004/assets/corpus/wiki-18.jsonl',
     'bytes': 14393579520, 'k1_sha256': corpus_sha, 'k4_sha256': corpus_sha,
     'k1_measurement_time': 1791354185.5714042, 'k4_measurement_time': 1791354168.6666956,
     'evidence': 'Both full decompressed files independently streaming-hashed in this subagent turn, captured tool outputs, no GPU.'}
]
sizes = {'.gitattributes':1477, '1_Pooling/config.json':200, 'README.md':67561,
         'config.json':650, 'model.safetensors':437955512, 'modules.json':387,
         'onnx/config.json':632, 'onnx/special_tokens_map.json':125,
         'onnx/tokenizer.json':711396, 'onnx/tokenizer_config.json':314,
         'onnx/vocab.txt':231508, 'openvino/openvino_model.xml':363139,
         'openvino/openvino_model_qint8_quantized.xml':663839,
         'sentence_bert_config.json':57, 'special_tokens_map.json':125,
         'tokenizer.json':711396, 'tokenizer_config.json':314, 'vocab.txt':231508}
for relative, sha in k1['encoder_sha256'].items():
    comparisons.append({'kind':'encoder', 'path':str(Path(k1['encoder_path']) / relative),
                        'k4_path':str(Path(k4['encoder_path']) / relative),
                        'bytes':sizes[relative], 'k1_sha256':sha,
                        'k4_sha256':k4['encoder_sha256'][relative],
                        'k1_measurement_time':k1['time'], 'k4_measurement_time':k4['time'],
                        'measurement_time_scope':'CPU encoder proof timestamps immediately following file SHA checks and model load; exact per-file hash instants were not recorded.',
                        'k1_evidence':'outputs/identity.json encoder_sha256, all fresh file hashes.',
                        'k4_evidence':'outputs/identity.json encoder_sha256, all fresh file hashes.'})
comparisons.append({'kind':'encoder_source', 'path':k1['original_module_path'],
                    'k4_path':k4['original_module_path'],
                    'k1_sha256':k1['original_module_sha256'],
                    'k4_sha256':k4['original_module_sha256'],
                    'k1_measurement_time':k1['time'], 'k4_measurement_time':k4['time'],
                    'measurement_time_scope':'Original module hashed in CPU identity proof construction.',
                    'evidence':'Both original CPU encoder identity.json receipts.'})
comparisons.append({'kind':'scanner_source',
                    'path':'/data/shaoshitong/search-e1-config-study-8gpu-20261004/source/experiments/search_e1/control/torch_streaming_flat.py',
                    'k4_path':'/weka_3/shaoshitong/search-e1-retriever-stream-batched-20261004/source/torch_streaming_flat.py',
                    'k1_sha256':scanner_sha, 'k4_sha256':scanner_sha,
                    'k1_measurement_time':1791355057.285379,
                    'k4_measurement_time':1791355057.2559173,
                    'evidence':'Parent reported matching fresh hashes; this subagent then independently read and SHA256-hashed both exact scanner source files, confirming the same value.'})
accepted = bool(k4_index and k4_index['accepted'] and all(
    row['k1_sha256'] == row['k4_sha256'] and row['k1_sha256'] for row in comparisons))
proof = {'accepted':accepted, 'time':time.time(), 'comparisons':comparisons,
         'scope':'Source file identity only. Does not prove candidate GPU scanner, memory, speed, or future trajectory determinism.',
         'k4_index_fresh_complete':bool(k4_index),
         'encoder_comparison':str(base / 'comparison.json'),
         'k1_encoder_receipt':'/data/shaoshitong/search-e1-retriever-analysis-20261007/cpu-encoder-identity/outputs/identity.json',
         'k4_encoder_receipt':'/weka_3/shaoshitong/search-e1-retriever-analysis-20261007/cpu-encoder-identity/outputs/identity.json'}
name = 'source-identity.json' if accepted else 'source-identity.partial.json'
(base / name).write_text(json.dumps(proof, indent=2))
print(json.dumps({'path':str(base / name), 'accepted':accepted, 'comparisons':len(comparisons)}))
