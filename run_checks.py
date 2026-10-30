# Copyright (c) 2026 Siyu Cai. SPDX-License-Identifier: MIT
"""Replay all seven verification suites in a temporary copy of the software."""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def temporary_workspace(parent):
    parent = parent.resolve()
    work = parent / ('correlation-replay-' + uuid4().hex)
    work.mkdir()
    try:
        yield work
    finally:
        if work.resolve().parent != parent or not work.name.startswith('correlation-replay-'):
            raise RuntimeError('Unexpected temporary directory location.')
        shutil.rmtree(work)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--compare', type=Path,
                        help='Directory of frozen results to compare byte for byte.')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists():
        raise SystemExit('Choose a new output directory; existing results are never overwritten.')
    inputs = ROOT / 'data/inputs'
    if not inputs.is_dir():
        raise SystemExit('Extract the dataset archive beside the software archive first.')
    commands = [
        ('extension', ['scripts/check_extension.py', '--output-dir', 'results/extension']),
        ('lower', ['scripts/check_lower.py', '--output-dir', 'results/lower']),
        ('growth', ['scripts/check_growth.py', '--output-dir', 'results/growth']),
        ('strengthening', ['check_strengthening.py', '--output', 'results/strengthening/checks.json']),
        ('moment', ['scripts/check_semantic_moment_oracle.py', '--output-dir', 'results/moment']),
        ('margin', ['scripts/check_semantic_margin.py', '--output-dir', 'results/margin']),
        ('hardness', ['scripts/check_whole_column_stopping_hardness.py', '--output-dir', 'results/hardness']),
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    with temporary_workspace(out.parent) as folder:
        work = Path(folder)
        for path in ROOT.rglob('*.py'):
            if any(part in ('.venv', '.git', 'results', 'artifacts') for part in path.relative_to(ROOT).parts):
                continue
            dest = work / path.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
        shutil.copytree(inputs, work/'data/inputs')
        (work/'results/strengthening').mkdir(parents=True)
        for name, command in commands:
            print('Running '+name, flush=True)
            result = subprocess.run([sys.executable, '-X', 'utf8', *command],
                                    cwd=work, capture_output=True, text=True, encoding='utf-8')
            if result.returncode:
                sys.stderr.write(result.stdout + result.stderr)
                raise SystemExit(result.returncode)
            print(name+': passed', flush=True)
        out.mkdir(parents=True)
        for path in sorted((work/'results').rglob('*')):
            if not path.is_file() or path.suffix not in ('.json', '.bp'):
                continue
            dest = out / path.relative_to(work/'results')
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
    actual = {p.relative_to(out).as_posix():digest(p) for p in sorted(out.rglob('*')) if p.is_file()}
    if args.compare:
        expected = {p.relative_to(args.compare).as_posix():digest(p)
                    for p in sorted(args.compare.rglob('*')) if p.is_file()}
        if actual != expected:
            changed = sorted(k for k in actual.keys() | expected.keys() if actual.get(k) != expected.get(k))
            raise SystemExit('Result mismatch: '+', '.join(changed))
        print('All '+str(len(actual))+' result files match the frozen bytes.', flush=True)
    print(json.dumps({'status':'passed','suites':len(commands),'result_files':len(actual)}, indent=2))


if __name__ == '__main__':
    main()
