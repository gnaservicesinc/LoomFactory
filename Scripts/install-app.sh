#!/bin/bash
set -euo pipefail
factory_root="$(cd "$(dirname "$0")/.." && pwd)"
factory_skip_build=false
factory_destdir=/
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-build) factory_skip_build=true; shift ;;
        --destdir)
            [[ $# -ge 2 ]] || { printf '%s\n' '--destdir requires a destination root.' >&2; exit 2; }
            factory_destdir="$2"; shift 2 ;;
        *) printf 'Unknown install argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done
if [[ "$factory_skip_build" != true ]]; then "$factory_root/Scripts/build-app.sh"; fi
python3 - "$factory_root/.build/application/Loom Factory.app" "$factory_destdir" <<'PY'
import datetime
import pathlib
import plistlib
import subprocess
import sys
import uuid

source = pathlib.Path(sys.argv[1])
installation_root = pathlib.Path(sys.argv[2]).expanduser().resolve()
destination = installation_root / 'Applications/Loom Factory.app'
destination.parent.mkdir(parents=True, exist_ok=True)
with (source / 'Contents/Info.plist').open('rb') as handle:
    metadata = plistlib.load(handle)
if metadata.get('CFBundleIdentifier') != 'com.gnaservicesinc.LoomFactory':
    raise SystemExit('The built app does not identify itself as Loom Factory.')
if metadata.get('LoomBuildConfiguration') != 'Release':
    raise SystemExit('Install requires an optimized Release build of Loom Factory.')
processes = subprocess.run(['/bin/ps', '-axo', 'command='], capture_output=True, text=True, check=True).stdout
if str(destination / 'Contents/MacOS/LoomFactory') in processes:
    raise SystemExit('Quit Loom Factory before installing this build.')
stage = destination.with_name(f'.Loom Factory.staging-{uuid.uuid4().hex}.app')
subprocess.run(['/usr/bin/ditto', str(source), str(stage)], check=True)
subprocess.run(['/usr/bin/codesign', '--verify', '--strict', '--verbose=2', str(stage)], check=True)
backup = None
if destination.exists():
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d-%H%M%S')
    backup_root = (pathlib.Path.home() / 'Library/Application Support/Loom Factory/Backups'
                   if installation_root == pathlib.Path('/') else installation_root / '.loom-suite-backups/Loom Factory')
    backup_root.mkdir(parents=True, exist_ok=True)
    backup = backup_root / f'Loom Factory.backup-{timestamp}-{uuid.uuid4().hex[:8]}.app'
    destination.rename(backup)
try:
    stage.rename(destination)
    subprocess.run(['/usr/bin/codesign', '--verify', '--strict', '--verbose=2', str(destination)], check=True)
except BaseException:
    if destination.exists():
        destination.rename(stage)
    if backup is not None:
        backup.rename(destination)
    raise
print(destination)
if backup is not None:
    print(f'Previous app retained at {backup}')
PY
