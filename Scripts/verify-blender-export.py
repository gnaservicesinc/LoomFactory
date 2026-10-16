"""Reopen a Test Model export in background Blender; no rendering/inference."""
import hashlib
import json
from pathlib import Path
import sys

import bpy


def verify(root):
    config = json.loads((root / 'review.json').read_text())
    bpy.ops.wm.open_mainfile(filepath=str(root / 'Material Review.blend'))
    assert len(bpy.data.scenes) == 9
    expected_planes = 2 if config['include_source'] else 1
    report = []
    expected_hashes = {r['sha256'] for r in [config['color'], *config['generated'].values(), *config['source'].values()]}
    packed = {hashlib.sha256(bytes(i.packed_file.data)).hexdigest() for i in bpy.data.images if i.packed_file}
    assert expected_hashes.issubset(packed)
    assert len(bpy.data.images) <= len(expected_hashes) + 1  # shared environment, no per-scene image copies
    for scene in bpy.data.scenes:
        assert scene.use_fake_user
        assert scene.render.threads_mode == 'AUTO'
        assert scene.cycles.pixel_filter_type == 'BLACKMAN_HARRIS'
        assert scene.cycles.filter_width == 1
        assert scene.cycles.device == 'CPU'
        assert scene.cycles.samples == 96
        assert scene.cycles.preview_dicing_rate == 8
        assert scene.cycles.use_pixel_jitter is False
        assert scene.eevee.fast_gi_ray_count == 16
        assert scene.eevee.fast_gi_step_count == 64
        assert scene.eevee.shadow_ray_count == 4
        assert scene.eevee.ray_tracing_options.resolution_scale == '1'
        assert next(n for n in scene.world.node_tree.nodes if n.type == 'BACKGROUND').inputs['Strength'].default_value == .5
        assert any(n.type == 'TEX_ENVIRONMENT' and n.image.packed_file for n in scene.world.node_tree.nodes)
        meshes = [o for o in scene.objects if o.type == 'MESH']
        assert len(meshes) == expected_planes
        objects = []
        for o in meshes:
            assert len(o.data.vertices) == 4 and len(o.data.polygons) == 1
            assert len(o.data.uv_layers) == 1 and o.data.uv_layers[0].name == 'UVMap'
            assert o.modifiers[0].subdivision_type == 'SIMPLE'
            if hasattr(o.modifiers[0], 'use_adaptive_subdivision'):
                assert o.modifiers[0].use_adaptive_subdivision
            mat = o.active_material
            assert mat['model_names'] == o['model_names']
            records = json.loads(o['exact_map_versions'])
            assert records == (config['source'] if o.name.startswith('Source material') else config['generated'])
            nodes = mat.node_tree.nodes
            for n in nodes:
                if n.type == 'TEX_IMAGE':
                    assert n.inputs['Vector'].links[0].from_node.type == 'MAPPING'
                    assert list(n.image.size) == [config['width'], config['height']]
                    assert n.image.colorspace_settings.name == ('sRGB' if n.name.startswith('color') else 'Non-Color')
                if n.type == 'NORMAL_MAP':
                    assert n.uv_map == 'UVMap'
                if n.type == 'DISPLACEMENT':
                    assert abs(n.inputs['Scale'].default_value - .025) < 1e-7
                    assert n.inputs['Midlevel'].default_value == .5
            objects.append({'name': o.name, 'material': mat.name, 'model_names': o['model_names'], 'vertices': 4})
        report.append({'scene': scene.name, 'engine': scene.render.engine, 'objects': objects})
    (root / 'reopen-verification.json').write_text(json.dumps({'scenes': report, 'verified_native_image_hashes': sorted(expected_hashes),
        'image_count': len(bpy.data.images), 'rendered': False}, indent=2))
    print('Verified nine scenes, exact packed maps, names, one UV per quad, optional second plane and render settings.')


if __name__ == '__main__':
    verify(Path(sys.argv[sys.argv.index('--')+1]).resolve())
