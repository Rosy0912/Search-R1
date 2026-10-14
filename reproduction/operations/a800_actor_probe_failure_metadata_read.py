"""One read of failed A9 metadata, sanitized error lines, and checkpoint inventory."""
import a800_actor_probe_gpu_ownership_read as reader

EXTRA = r'''
    stage='terminal_existing_metadata'
    terminal=dict(files=[],errors=[],checkpoint_directories=[],all_registered_births_exited=True)
    for job in out['jobs']:
        for record in job['registered_identities']:
            current=record['after']
            gone=(current['read_status']=='absent_or_exited' or
                  (current['read_status']=='read' and
                   (current['start_ticks']!=record['start_ticks'] or current['state'] in ('Z','X','x'))))
            if not gone:terminal['all_registered_births_exited']=False
    def existing_bytes(relative,limit):
        need(relative.startswith('logs/') or relative.startswith('checkpoints/'),'terminal_path_scope')
        path=p['base']+'/'+relative
        directory,name=path.rsplit('/',1)
        directory_fd=open_directory(directory)
        try:
            fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=directory_fd)
            try:
                st=os.fstat(fd);need(stat.S_ISREG(st.st_mode) and st.st_uid==os.getuid() and st.st_size<=limit,'terminal_file_type_owner_limit')
                pieces=[];count=0
                while True:
                    chunk=os.read(fd,min(65536,limit+1-count))
                    if not chunk:break
                    pieces.append(chunk);count+=len(chunk);need(count<=limit,'terminal_file_limit')
                need(stable_stat(st)==stable_stat(os.fstat(fd)),'terminal_file_changed')
            finally:os.close(fd)
        finally:os.close(directory_fd)
        raw=b''.join(pieces)
        terminal['files'].append(dict(relative=relative,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
        return raw
    for relative in ('logs/cleanup-status.json','logs/pipeline-status.json','logs/actual-updates.jsonl','logs/actual-attempts.jsonl'):
        raw=existing_bytes(relative,2*1024*1024)
        if relative.endswith('.json'):
            value=json.loads(raw)
            terminal[relative.split('/')[-1]]=value
        else:
            need(raw.endswith(b'\n'),'terminal_scalar_partial_row')
            rows=[json.loads(line) for line in raw.splitlines()]
            need(all(row['run_id']==p['run_id'] for row in rows),'terminal_scalar_identity')
            terminal[relative.split('/')[-1]]=rows
    for relative in ('logs/trainer.log','logs/search-g1-common8-training.log'):
        raw=existing_bytes(relative,2*1024*1024)
        for original in raw.decode('utf-8',errors='replace').splitlines():
            line=re.sub(r'\x1b\[[0-9;]*m','',original)
            line=re.sub(r'^\([^)]*pid=[0-9]+[^)]*\)\s*','',line).strip()
            if (re.match(r'^(?:torch\.(?:cuda\.)?)?OutOfMemoryError:',line) or
                re.match(r'^RuntimeError:\s*(?:CUDA|.*out of memory)',line,re.I) or
                line.startswith('torch.OutOfMemoryError:') or
                line.startswith('CUDA out of memory')):
                terminal['errors'].append(dict(log=relative,line=line[:4096]))
        need(len(terminal['errors'])<=64,'terminal_error_count')
    checkpoint_root=p['base']+'/checkpoints/search-g1-common8/actor'
    try:
        fd=open_directory(checkpoint_root)
        try:
            names=os.listdir(fd);need(len(names)<=256,'checkpoint_inventory_count')
            for name in sorted(names):
                if not re.fullmatch('global_step_[0-9]+',name):continue
                info=os.stat(name,dir_fd=fd,follow_symlinks=False)
                need(stat.S_ISDIR(info.st_mode) and info.st_uid==os.getuid(),'checkpoint_directory_type_owner')
                cfd=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
                try:
                    children=os.listdir(cfd);need(len(children)<=256,'checkpoint_files_count')
                    sizes=[]
                    for child in sorted(children):
                        info=os.stat(child,dir_fd=cfd,follow_symlinks=False)
                        sizes.append(dict(name=child,bytes=info.st_size,regular=stat.S_ISREG(info.st_mode),directory=stat.S_ISDIR(info.st_mode),symlink=stat.S_ISLNK(info.st_mode)))
                    terminal['checkpoint_directories'].append(dict(name=name,files=sizes))
                finally:os.close(cfd)
        finally:os.close(fd)
    except FileNotFoundError:terminal['checkpoint_root_absent']=True
    terminal['owned_cuda_apps']=[app for app in out['compute_apps'] if app['classification']=='owned']
    terminal['no_model_or_optimizer_payload_read']=True
    out['terminal_evidence']=terminal
'''

if __name__ == '__main__':
    marker = "except BaseException as error:\n    # Never print source JSON"
    assert reader.REMOTE_BODY.count(marker) == 1
    reader.REMOTE_BODY = reader.REMOTE_BODY.replace(marker, EXTRA + "\n" + marker)
    raise SystemExit(reader.main())
