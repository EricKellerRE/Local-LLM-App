"""Deploy the case-free source bundle into the current user's Documents."""
import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import time
import venv
import zipfile

def run(command, **kwargs):
    print("Running:", " ".join(map(str,command)), flush=True)
    return subprocess.run(list(map(str,command)),check=True,**kwargs)

def overlay(bundle, destination, backup):
    changed=0
    with zipfile.ZipFile(bundle) as archive:
        for member in archive.infolist():
            relative=PurePosixPath(member.filename)
            if member.is_dir() or not relative.parts or relative.parts[0] not in {'Local LLM','Grid-Workshop'}:
                continue
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError("Unsafe archive path")
            target=destination.joinpath(*relative.parts)
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError("Archive path escapes destination")
            if target.exists() and (target.name=='.env' or target.name=='app-settings.json'):
                continue
            payload=archive.read(member)
            if target.exists():
                if target.read_bytes()==payload: continue
                prior=backup.joinpath(*relative.parts)
                prior.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(target,prior)
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(payload)
            changed+=1
    return changed

def update_env(path, settings):
    lines=path.read_text(encoding='utf-8-sig').splitlines() if path.exists() else []
    output=[]
    seen=set()
    for line in lines:
        key=line.split('=',1)[0].strip() if '=' in line and not line.lstrip().startswith('#') else ''
        if key in settings:
            output.append(key+'='+settings[key]);seen.add(key)
        else: output.append(line)
    output.extend(key+'='+value for key,value in settings.items() if key not in seen)
    path.write_text('\n'.join(output)+'\n',encoding='utf-8')

def main():
    if sys.version_info < (3,10): raise RuntimeError('Python 3.10 or newer is required.')
    staging=Path(__file__).resolve().parent
    documents=Path(os.environ['USERPROFILE'])/'Documents'
    app=documents/'Local LLM'
    grid=documents/'Grid-Workshop'
    stamp=time.strftime('%Y%m%d-%H%M%S')
    backup=documents/('Local-Grid-source-backup-'+stamp)
    for name,expected in json.loads((staging/'SHA256.json').read_text()).items():
        if hashlib.sha256((staging/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Checksum mismatch: '+name)
    changed=overlay(staging/'source.zip',documents,backup)
    print('Source files installed:',changed,'Backups:',backup,flush=True)
    model=documents/'gemma-4-26B-A4B-it-GGUF'/'gemma-4-26B-A4B-it-Q4_K_M.gguf'
    if not model.is_file(): raise FileNotFoundError('Expected Gemma GGUF is missing: '+str(model))
    cuda_base=Path(os.environ.get('ProgramFiles',r'C:\Program Files'))/'NVIDIA GPU Computing Toolkit'/'CUDA'
    candidates=sorted(cuda_base.glob('v12*/bin'),reverse=True)
    cuda_bin=next((p for p in candidates if (p/'cudart64_12.dll').is_file() and (p/'cublas64_12.dll').is_file()),None)
    if cuda_bin is None:
        raise RuntimeError('CUDA 12 toolkit DLLs were not found. No system CUDA installation was changed.')
    runtime=app/'runtimes'/'llama-b12144-cuda12'
    # Own versioned runtime directory only; preserve any other existing runtimes.
    runtime.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(staging/'llama-b12144-cuda12.zip') as archive:
        for member in archive.infolist():
            if member.is_dir(): continue
            rel=PurePosixPath(member.filename)
            if rel.is_absolute() or '..' in rel.parts: raise ValueError('Unsafe runtime member')
            target=runtime.joinpath(*rel.parts)
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(archive.read(member))
    environment=os.environ.copy()
    environment['PATH']=str(cuda_bin)+os.pathsep+environment.get('PATH','')
    devices=run([runtime/'llama-server.exe','--list-devices'],env=environment,capture_output=True,text=True)
    device_text=devices.stdout+devices.stderr
    print(device_text,flush=True)
    if not re.search(r'CUDA\d+\s*:', device_text):
        raise RuntimeError('CUDA runtime did not report any CUDA device; refusing CPU fallback.')
    update_env(app/'.env',{
        'LOCAL_MODEL_ID':str(model),'LOCAL_MODEL_KIND':'gguf','LOCAL_MODEL_DEVICE':'cuda',
        'LOCAL_LLAMA_SERVER':str(runtime/'llama-server.exe'),'LOCAL_CUDA_BIN':str(cuda_bin),
        'GRID_WORKSHOP_ROOT':str(grid),'LOCAL_MODEL_CONTEXT_WINDOW':'16384','LOCAL_MODEL_MAX_NEW_TOKENS':'2048',
        'LOCAL_MODEL_REASONING_BUDGET':'0','LOCAL_MODEL_EAGER_LOAD':'false','LOCAL_ROUTER_MODEL_ID':'',
        'LOCAL_RESEARCH_CLASSIFIER_MODEL_ID':'','LOCAL_TOOL_ACTION_MAX_NEW_TOKENS':'1024',
        'LOCAL_POST_TOOL_DECISION_MAX_NEW_TOKENS':'2048'})
    config=app/'config'/'app-settings.json'
    settings=json.loads(config.read_text()) if config.exists() else {}
    settings['models_directory']=str(model.parent)
    settings.setdefault('data_directory','data')
    config.write_text(json.dumps(settings,indent=2),encoding='utf-8')
    python=app/'.venv'/'Scripts'/'python.exe'
    if not python.is_file(): venv.EnvBuilder(with_pip=True).create(app/'.venv')
    run([python,'-m','pip','install','--upgrade','pip'],cwd=app)
    run([python,'-m','pip','install','-r',app/'requirements.txt'],cwd=app)
    run([python,'-m','pip','install','-e',grid/'powerworld-aux-agent'],cwd=app)
    run([python,app/'scripts'/'grid_analysis_readiness.py'],cwd=app,env=environment)
    print('Readiness process exited. Starting Gemma synthetic agent test; initial model loading can take several minutes.',flush=True)
    smoke=subprocess.run([str(python),'-u',str(app/'scripts'/'gguf_workflow_smoke.py')],cwd=app,env=environment)
    report={'installed':True,'source_files_changed':changed,'backup':str(backup),'model':str(model),
            'cuda_bin':str(cuda_bin),'devices':device_text,'synthetic_agent_test_exit_code':smoke.returncode}
    (staging/'DEPLOYMENT-RESULT.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    subprocess.Popen([str(python.with_name('pythonw.exe')),str(app/'launcher.py')],cwd=app)
    print('App opened. Synthetic test exit code:',smoke.returncode,flush=True)
    return smoke.returncode

if __name__=='__main__':
    try:
        raise SystemExit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        raise SystemExit(1)
