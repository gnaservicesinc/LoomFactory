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
/usr/bin/xcrun swift "$factory_root/Scripts/InstallApp.swift" "$factory_root/.build/application/Loom Factory.app" "$factory_destdir"
