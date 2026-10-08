import importlib.util,subprocess,pathlib,json,time,shlex
chat=pathlib.Path(__file__).resolve().parents[2];here=pathlib.Path(__file__).parent;spec=importlib.util.spec_from_file_location('remote',chat/'work/remote.py');remote=importlib.util.module_from_spec(spec);spec.loader.exec_module(remote)
own='/dataset/shaoshitong/search-r1-rl/tree-grpo';library=own+'/cpu-blas-experiment/usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblas.so.0'
program='/normal symbol .sgemm_.$/ || /_swigfaiss.*normal symbol .openblas_get_num_threads.$/'
body='/usr/bin/nice -n 19 env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=\'\' OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 LD_PRELOAD='+library+' LD_DEBUG=bindings '+own+'/env/bin/python - 2> >(/usr/bin/awk '+shlex.quote(program)+' >&2)'
command='/usr/bin/timeout 120 bash -c '+shlex.quote(body);start=time.perf_counter();r=subprocess.run(remote.route(command),input=(here/'microbenchmark-with-binding.py').read_bytes(),capture_output=True,timeout=180)
(here/'preload-retry-stdout.txt').write_bytes(r.stdout);(here/'preload-retry-sgemm-bindings.txt').write_bytes(r.stderr)
d=json.loads((here/'isolated-microbenchmark-results.json').read_text());d['preload_retry']={'returncode':r.returncode,'remote_wall_seconds':time.perf_counter()-start,'binding_lines':r.stderr.decode(errors='replace').splitlines()}
if r.returncode==0:d['preload_retry']['measurement']=json.loads(r.stdout)
(here/'isolated-microbenchmark-results.json').write_text(json.dumps(d,indent=2));print('preload retry',r.returncode,flush=True)
