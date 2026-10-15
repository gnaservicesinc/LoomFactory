"""Drive the shared Cycles renderer for human artistic review, without training.

Blender --background --factory-startup --disable-autoexec --threads 4
--python render-art-direction-review.py -- configuration.json
"""
import array
import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import bpy


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


path = Path(sys.argv[sys.argv.index('--') + 1]).resolve()
config = json.loads(path.read_text())
module_path = Path(__file__).with_name('render-material-comparison.py')
spec = importlib.util.spec_from_file_location('material_review', module_path)
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)
out = path.parent / 'renders'
out.mkdir(exist_ok=True)
inputs = {config['environment']: sha(config['environment'])}
records = []
for material in config['materials']:
    record = json.loads(Path(material['record']).read_text())
    native_dimensions = None
    for key in ['input', 'depth', 'normal', 'roughness']:
        inputs[record[key]] = sha(record[key])
        dimensions = tuple(renderer.load_image(record[key], key != 'input').size)
        if max(dimensions) != 1024:
            raise ValueError('Artistic review requires existing native 1K maps; choose the 1K material instead of resizing a larger map.')
        if native_dimensions is not None and dimensions != native_dimensions:
            raise ValueError('The 1K color, height, normal and roughness maps must have matching native dimensions.')
        native_dimensions = dimensions
    image = renderer.load_image(record['depth'], True)
    pixels = array.array('f', [0]) * len(image.pixels)
    image.pixels.foreach_get(pixels)
    values = pixels[::image.channels]
    assert values and all(math.isfinite(v) for v in values)
    records.append((material, record, math.fsum(values) / len(values)))

for preset in config['presets']:
    assert 0 < preset['height_gain'] <= 16
    assert 0 <= preset['normal_strength'] <= 8
    assert preset.get('roughness_constant') is None or 0 <= preset['roughness_constant'] <= 1

completed, assets = [], []
views = config.get('views', [False, True])
lights = config.get('lights', ['studio', 'rotated', 'grazing'])
total = len(records) * len(config['presets']) * len(lights) * len(views)
for material, record, midpoint in records:
    directory = out / material['id']
    directory.mkdir(exist_ok=True)
    model_dir = path.parent / 'models' / material['id']
    model_dir.mkdir(parents=True, exist_ok=True)
    exported_gains = set()
    for preset in config['presets']:
        for light in lights:
            for neutral in views:
                name = f'{preset["id"]}-{light}{"-neutral" if neutral else ""}.png'
                result = directory / name
                if result.exists():
                    raise FileExistsError('Preserve existing review: ' + str(result))
                options = {**preset, 'height_midpoint': midpoint, 'pixels_per_node': config['pixels_per_node'], 'max_mesh_edge_vertices': config.get('max_mesh_edge_vertices', 513), 'environment': config['environment'], 'environment_rotation': 135 if light == 'rotated' else 0, 'environment_strength': 0.35 if light == 'grazing' else 3}
                if not neutral and preset['height_gain'] not in exported_gains and preset['height_gain'] in [1, 4]:
                    options['export_obj'] = str(model_dir / f'height-{preset["height_gain"]}x.obj')
                    exported_gains.add(preset['height_gain'])
                if preset['id'] == 'relief4' and light == lights[0] and not neutral:
                    options['save_blend'] = str(model_dir / 'editable-material.blend')
                exported = renderer.render_set(record, '', result, light, neutral, options)
                if exported:
                    assert exported['vertices'] > 0 and exported['z_range'][1] > exported['z_range'][0]
                    assert sha(exported['diffuse']) == sha(record['input'])
                    assets.append(exported)
                completed.append({'material': material['id'], 'preset': preset['id'], 'light': light, 'neutral': neutral, 'output': str(result), 'sha256': sha(result), 'height_midpoint': midpoint})
                (path.parent / 'render-status.json').write_text(json.dumps({'state': 'rendering', 'completed': len(completed), 'total': total, 'last': completed[-1]}, indent=2))
                print('ART_REVIEW', len(completed), '/', total, material['id'], name, flush=True)
for name, checksum in inputs.items():
    assert sha(name) == checksum, 'Input changed during review: ' + name
receipt = {'schema': 'loom-art-direction-render-v1', 'completed_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'configuration_sha256': sha(path), 'script_sha256': sha(__file__), 'shared_renderer_sha256': sha(module_path), 'cpu_only': True, 'training_started': False, 'source_maps_written': False, 'model_selection_changed': False, 'input_sha256_unchanged': inputs, 'results': completed, 'mesh_assets': assets, 'height_application': '0.08 * height_gain * (saved_height - mean(saved_height)); midpoint only removes uniform surface translation. No target matching or histogram remapping.', 'normal_application': 'OpenGL tangent Normal Map node strength, or explicitly disabled for geometry-only control. No independent RGB contrast.', 'roughness': 'Saved prediction or labeled constant control; no required correlation with color, depth or normal.', 'acceptance': 'Human judgment of usable, visually pleasing surfaces for the color map. Reference metrics are diagnostics, not a pass/fail score. Authored map differences are allowed.'}
(path.parent / 'render-receipt.json').write_text(json.dumps(receipt, indent=2))
(path.parent / 'render-status.json').write_text(json.dumps({'state': 'completed', 'completed': len(completed), 'total': total}, indent=2))
