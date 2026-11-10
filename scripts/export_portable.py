"""Build a Windows transfer ZIP without machine-specific environments or secrets."""
import argparse
import json
import re
import subprocess
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {'.git', '.venv', 'localmodel-env', 'models', 'data', 'runs', 'deliverables',
            'portable-dist', 'readiness_patch', '__pycache__', 'node_modules', '.pytest_cache'}
CASE_TYPES = {'.pwb', '.pww', '.raw', '.sav', '.xlsx', '.xls', '.csv', '.log'}


def source_files(root):
    result = subprocess.run(['git', '-C', str(root), 'ls-files', '-z', '--cached', '--others', '--exclude-standard'],
                            check=True, capture_output=True)
    for name in sorted(set(result.stdout.decode().split('\0')) - {''}):
        path = root / name
        if (not path.is_file() or path.is_symlink() or any(part in EXCLUDED for part in Path(name).parts)
                or path.name.startswith('.env') or path.name == 'app-settings.json'
                or path.suffix.lower() in CASE_TYPES):
            continue
        yield path


def model_shards(path):
    path = path.resolve()
    if not path.is_file() or path.suffix.lower() != '.gguf' or path.name.startswith('mmproj'):
        raise ValueError(f'Not a language model GGUF: {path}')
    match = re.search(r'-(\d{5})-of-(\d{5})\.gguf$', path.name)
    if not match:
        return [path]
    count = int(match[2])
    prefix = path.name[:match.start()]
    shards = [path.with_name(f'{prefix}-{i:05d}-of-{count:05d}.gguf') for i in range(1, count+1)]
    if not all(shard.is_file() for shard in shards):
        raise ValueError(f'Missing split GGUF shard for {path}')
    return shards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'portable-dist/Local-Model-Windows.zip')
    parser.add_argument('--grid-root', type=Path, default=ROOT.parent/'Grid-Workshop')
    parser.add_argument('--model', action='append', type=Path, default=[])
    parser.add_argument('--include-data', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite existing transfer: {args.output}')
    if args.include_data:
        import socket
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1', 8765)) == 0:
                raise RuntimeError('Close Local Model before exporting chat data.')
    files = [(path, 'Local LLM/'+path.relative_to(ROOT).as_posix()) for path in source_files(ROOT)]
    if args.grid_root.is_dir():
        files += [(path, 'Grid-Workshop/'+path.relative_to(args.grid_root).as_posix()) for path in source_files(args.grid_root)]
        catalog = args.grid_root/'powerworld-aux-agent/data/parsed/source_catalogs'
        if not (catalog/'fields.json').is_file():
            raise FileNotFoundError('Grid Workshop field catalog is missing; transfer would not support field discovery.')
        files += [(path, 'Grid-Workshop/'+path.relative_to(args.grid_root).as_posix()) for path in catalog.glob('*.json')]
    runtime = ROOT/'runtimes/llama-b12144'
    if not (runtime/'llama-server.exe').is_file():
        raise FileNotFoundError('Standalone CPU runtime is missing.')
    files += [(path, 'Local LLM/'+path.relative_to(ROOT).as_posix()) for path in runtime.rglob('*') if path.is_file()]
    models = []
    for model in args.model:
        for shard in model_shards(model):
            destination = 'Local LLM/models/'+shard.name
            if destination not in {name for _, name in files}:
                files.append((shard, destination))
        models.append('models/'+model_shards(model)[0].name)
    if args.include_data:
        from local_model_app.app_settings import AppSettingsStore
        data = AppSettingsStore(ROOT).paths().data_directory
        files += [(path, 'Local LLM/data/'+path.relative_to(data).as_posix()) for path in data.rglob('*') if path.is_file() and not path.is_symlink()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    env = '\n'.join(['LOCAL_MODEL_ID='+ (models[0] if models else ''), 'LOCAL_MODEL_KIND=gguf',
                     'LOCAL_MODEL_DEVICE=cpu', 'LOCAL_MODEL_CONTEXT_WINDOW=16384', 'LOCAL_MODEL_MAX_NEW_TOKENS=2048',
                     'LOCAL_MODEL_REASONING_BUDGET=0', 'LOCAL_MODEL_EAGER_LOAD=false', 'LOCAL_ROUTER_MODEL_ID=',
                     'LOCAL_RESEARCH_CLASSIFIER_MODEL_ID=', 'GRID_WORKSHOP_ROOT=../Grid-Workshop',
                     'LOCAL_LLAMA_SERVER=runtimes/llama-b12144/llama-server.exe', ''])
    with zipfile.ZipFile(args.output, 'x', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for path, name in files:
            archive.write(path, name, compress_type=zipfile.ZIP_STORED if path.suffix == '.gguf' else zipfile.ZIP_DEFLATED)
        archive.writestr('Local LLM/.env', env)
        archive.writestr('Local LLM/config/app-settings.json', json.dumps({'data_directory':'data','models_directory':'models'}))
        archive.writestr('TRANSFER-INVENTORY.json', json.dumps({'files':len(files),'models':models,'chat_data_included':args.include_data,
            'excluded':['credentials and original .env', 'Python environments', 'case/workbook/CSV/log source files', 'generated runs and deliverables'],
            'instructions':'Local LLM/PORTABLE_WINDOWS.md'}, indent=2))
    print(json.dumps({'archive':str(args.output),'files':len(files),'bytes':args.output.stat().st_size}))


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(ROOT))
    main()
