"""Small immutable source snapshots accompanying research runs."""
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib
import json
from ragqa.paths import PROJECT_ROOT
from .verify_dataset import digest


def source_snapshot(destination):
    if destination.exists():
        return digest(destination)
    patterns = ('experiments/*.py', 'ragqa/*.py', 'configs/*.json', 'requirements/*.txt')
    with ZipFile(destination, 'w', compression=ZIP_DEFLATED) as archive:
        for pattern in patterns:
            for path in sorted(PROJECT_ROOT.glob(pattern)):
                archive.write(path, path.relative_to(PROJECT_ROOT).as_posix())
    return digest(destination)


def records_sha256(records):
    """Platform-independent record identity (including text), separate from file SHA."""
    hasher = hashlib.sha256()
    for row in records:
        hasher.update(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                 separators=(',', ':')).encode('utf8'))
        hasher.update(b'\n')
    return hasher.hexdigest()
