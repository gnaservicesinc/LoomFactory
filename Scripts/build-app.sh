#!/bin/bash
set -euo pipefail

factory_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$factory_root"

swift build --configuration release
factory_binary_dir="$(swift build --configuration release --show-bin-path)"
factory_bundle="$factory_root/.build/application/Loom Factory.app"
factory_stage="$factory_root/.build/application/Loom Factory.staging.app"

if [[ -e "$factory_stage" ]]; then
    /bin/rm -rf "$factory_stage"
fi
mkdir -p "$factory_stage/Contents/MacOS" "$factory_stage/Contents/Resources"
cp "$factory_binary_dir/LoomFactory" "$factory_stage/Contents/MacOS/LoomFactory"
cp Resources/Info.plist "$factory_stage/Contents/Info.plist"
/usr/libexec/PlistBuddy -c 'Add :LoomBuildConfiguration string Release' "$factory_stage/Contents/Info.plist"
swift Scripts/MakeIcon.swift "$factory_stage/Contents/Resources"
/usr/bin/ditto Resources/ModelLicenses "$factory_stage/Contents/Resources/ModelLicenses"
cp Sources/LoomCore/Resources/BlenderExport.py "$factory_stage/Contents/Resources/BlenderExport.py"
factory_bootstrap="${LOOM_FACTORY_BOOTSTRAP_MODEL:-$factory_root/Resources/DTTC3.mlmodel}"
if [[ ! -e "$factory_bootstrap" ]]; then
    printf 'Categorizer asset missing: %s\n' "$factory_bootstrap" >&2
    exit 1
fi
case "$factory_bootstrap" in
    *.mlmodelc)
        /usr/bin/ditto "$factory_bootstrap" "$factory_stage/Contents/Resources/DTTC3.mlmodelc"
        ;;
    *.mlpackage)
        /usr/bin/ditto "$factory_bootstrap" "$factory_root/.build/DTTC3.mlpackage"
        /usr/bin/xcrun coremlcompiler compile "$factory_root/.build/DTTC3.mlpackage" "$factory_stage/Contents/Resources"
        ;;
    *.mlmodel)
        cp "$factory_bootstrap" "$factory_root/.build/DTTC3.mlmodel"
        /usr/bin/xcrun coremlcompiler compile "$factory_root/.build/DTTC3.mlmodel" "$factory_stage/Contents/Resources"
        ;;
    *)
        printf 'Choose a CoreML categorizer (.mlmodel, .mlpackage, or .mlmodelc).\n' >&2
        exit 1
        ;;
esac
factory_map_classifier="$factory_root/Resources/MatTypeClassifier.mlmodel"
if [[ ! -f "$factory_map_classifier" ]]; then
    printf 'Map-type classifier asset missing: %s\n' "$factory_map_classifier" >&2
    exit 1
fi
/usr/bin/xcrun coremlcompiler compile "$factory_map_classifier" "$factory_stage/Contents/Resources"
"$factory_root/Scripts/verify-native-bundle.sh" "$factory_stage"
codesign --force --sign - --options runtime "$factory_stage"
codesign --verify --strict --verbose=2 "$factory_stage"
if [[ -e "$factory_bundle" ]]; then
    factory_previous="$factory_root/.build/application/Loom Factory.previous.app"
    if [[ -e "$factory_previous" ]]; then
        /bin/rm -rf "$factory_previous"
    fi
    /bin/mv "$factory_bundle" "$factory_previous"
fi
/bin/mv "$factory_stage" "$factory_bundle"
if [[ -n "${factory_previous:-}" && -e "$factory_previous" ]]; then
    /bin/rm -rf "$factory_previous"
fi
printf '%s\n' "$factory_bundle"
