#!/bin/bash
set -euo pipefail
factory_root="$(cd "$(dirname "$0")/.." && pwd)"
"$factory_root/Scripts/build-app.sh"
open "$factory_root/.build/application/Loom Factory.app"
