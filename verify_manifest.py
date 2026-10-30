# Copyright (c) 2026 Siyu Cai. SPDX-License-Identifier: MIT
"""Verify both archive parts and reject missing, modified, or extra release files."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IGNORED = {'.git', '.venv', '__pycache__', 'results', 'artifacts'}


def check(part, base, manifest):
    document = json.loads(manifest.read_text(encoding='utf-8'))
    assert document['part'] == part and document['version'] == '1.0.0'
    expected = document['files']
    actual = {}
    for path in sorted(base.rglob('*')):
        rel = path.relative_to(base)
        if not path.is_file() or path == manifest:
            continue
        if any(p in IGNORED for p in rel.parts) or (part == 'software' and rel.parts[0] == 'data'):
            continue
        content = path.read_bytes()
        actual[rel.as_posix()] = {'size':len(content),'sha256':hashlib.sha256(content).hexdigest()}
    if actual != expected:
        changed = sorted(k for k in actual.keys() | expected.keys() if actual.get(k) != expected.get(k))
        raise SystemExit(part+' manifest mismatch: '+', '.join(changed))
    return len(actual)


if __name__ == '__main__':
    counts = {part:check(part,base,base/'manifest.json')
              for part,base in (('software',ROOT),('dataset',ROOT/'data'))}
    print(json.dumps({'status':'passed','verified_files':counts},indent=2))
