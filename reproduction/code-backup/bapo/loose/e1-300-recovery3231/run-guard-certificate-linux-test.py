import subprocess,sys,shlex,pathlib
script=pathlib.Path('work/e1-300-recovery3231/guard-certificate-linux-test.py').read_text()
command='python3 -c '+shlex.quote(script)
raise SystemExit(subprocess.run([sys.executable,'work/h100_exec.py','--socks','--timeout','120','--stdin-file','work/e1-300-recovery3231/guard-certificate-linux-payload.json',command]).returncode)
