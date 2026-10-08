import importlib.util,subprocess,pathlib,json,time
chat=pathlib.Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('remote',chat/'work/remote.py');remote=importlib.util.module_from_spec(spec);spec.loader.exec_module(remote)
script=(pathlib.Path(__file__).with_name('microbenchmark-with-binding.py')).read_bytes();own='/dataset/shaoshitong/search-r1-rl/tree-grpo';library=own+'/cpu-blas-experiment/usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblas.so.0';results={}
for mode in ['default','preload']:
 env='env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES=\'\' OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 LD_DEBUG=bindings '
 if mode=='preload':env+='LD_PRELOAD='+library+' '
 command='/usr/bin/timeout 120 /usr/bin/nice -n 19 '+env+own+'/env/bin/python -'
 start=time.perf_counter();result=subprocess.run(remote.route(command),input=script,capture_output=True,timeout=180)
 here=pathlib.Path(__file__).parent;(here/(mode+'-stdout.txt')).write_bytes(result.stdout);(here/(mode+'-loader-bindings-full.txt')).write_bytes(result.stderr)
 selected=[line for line in result.stderr.decode(errors='replace').splitlines() if 'sgemm_' in line or ('_swigfaiss' in line and 'openblas_' in line)]
 (here/(mode+'-sgemm-bindings.txt')).write_text('\n'.join(selected)+'\n')
 results[mode]={'returncode':result.returncode,'remote_wall_seconds':time.perf_counter()-start,'binding_lines':selected}
 if result.returncode==0:results[mode]['measurement']=json.loads(result.stdout)
 (here/'isolated-microbenchmark-results.json').write_text(json.dumps(results,indent=2))
 print(mode,result.returncode,'completed',flush=True)
