"""Reopen packed Blender deliverables and verify their template/map contract."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import bpy
def numeric_props(owner):
    result = {}
    for prop in owner.bl_rna.properties:
        if prop.type not in ['BOOLEAN', 'INT', 'FLOAT', 'ENUM', 'STRING'] or prop.identifier == 'rna_type':
            continue
        try:
            value = getattr(owner, prop.identifier)
            result[prop.identifier] = list(value) if getattr(prop, 'is_array', False) else value
        except (AttributeError, TypeError):
            pass
    return result


def mesh_snapshot(obj):
    return {'vertices': [list(v.co) for v in obj.data.vertices],
            'polygons': [list(p.vertices) for p in obj.data.polygons],
            'uv': [list(d.uv) for d in obj.data.uv_layers[0].data],
            'scale': list(obj.scale), 'modifiers': [numeric_props(m) for m in obj.modifiers]}


def verify(folder):
    receipt = json.loads((folder / 'build-receipt.json').read_text())
    template = Path(receipt['template'])
    assert hashlib.sha256(template.read_bytes()).hexdigest() == receipt['template_sha256']
    module = Path(__file__).with_name('build-blender-template-review.py')
    spec = importlib.util.spec_from_file_location('template_builder', module)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    bpy.ops.wm.open_mainfile(filepath=str(template))
    source = bpy.context.scene
    plane = next(o for o in source.objects if o.type == 'MESH')
    geometry = mesh_snapshot(plane)
    cycles = numeric_props(source.cycles)
    eevee = numeric_props(source.eevee)
    raytracing = numeric_props(source.eevee.ray_tracing_options)
    view = numeric_props(source.view_settings)
    template_nodes, _, _, template_normal, _ = builder.mapped_nodes(plane.active_material)
    textures = {role: numeric_props(node) for role, node in template_nodes.items()}
    mapping = next(n for n in plane.active_material.node_tree.nodes if n.type == 'MAPPING')
    mapping_inputs = {s.name: list(s.default_value) for s in mapping.inputs if hasattr(s, 'default_value')}
    checked = []
    for entry in receipt['results']:
        path = Path(entry['file'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']
        bpy.ops.wm.open_mainfile(filepath=str(path))
        assert len(bpy.data.scenes) == 9
        assert bpy.context.scene.name == '01 Cycles - Combined'
        for scene in bpy.data.scenes:
            surface = next(o for o in scene.objects if o.type == 'MESH')
            actual = mesh_snapshot(surface)
            # Modifier runtime identifiers and evaluation times may change on reload.
            for record in actual['modifiers'] + geometry['modifiers']:
                for key in ['persistent_uid', 'execution_time', 'is_active',
                            'open_adaptive_subdivision_panel', 'open_advanced_panel']:
                    record.pop(key, None)
            assert actual == geometry, (entry['id'], scene.name, 'geometry/modifier changed')
            assert scene.render.threads_mode == 'AUTO' and scene.render.threads == os.cpu_count()
            assert scene.cycles.pixel_filter_type == 'BLACKMAN_HARRIS' and scene.cycles.filter_width == 1
            assert numeric_props(scene.view_settings) == view
            assert numeric_props(scene.eevee) == eevee
            assert numeric_props(scene.eevee.ray_tracing_options) == raytracing
            actual_cycles = numeric_props(scene.cycles)
            for key in ['use_denoising', 'use_preview_denoising']:
                if scene['review_control'] == 'raw':
                    assert actual_cycles[key] is False
                    actual_cycles[key] = cycles[key]
            assert actual_cycles == cycles
            assert all(abs(n.inputs['Strength'].default_value - .05) < 1e-7
                       for n in scene.world.node_tree.nodes if n.type == 'BACKGROUND')
            material = surface.active_material
            assert material.displacement_method == 'BOTH'
            normal = next(n for n in material.node_tree.nodes if n.type == 'NORMAL_MAP')
            displacement = next(n for n in material.node_tree.nodes if n.type == 'DISPLACEMENT')
            assert (normal.uv_map, normal.space, normal.convention, normal.base) == ('UVMap', 'TANGENT', 'OPENGL', 'DISPLACED')
            assert abs(displacement.inputs['Scale'].default_value - .025) < 1e-7
            assert displacement.inputs['Midlevel'].default_value == .5
            mapping = next(n for n in material.node_tree.nodes if n.type == 'MAPPING')
            assert {s.name: list(s.default_value) for s in mapping.inputs if hasattr(s, 'default_value')} == mapping_inputs
            mapped = [n for n in material.node_tree.nodes if n.type == 'TEX_IMAGE']
            assert len(mapped) == 4
            expected = {r['sha256'] for r in entry['maps'].values()}
            found = set()
            for node in mapped:
                assert node.inputs['Vector'].links[0].from_node == mapping
                image = node.image
                assert tuple(image.size) == (1024, 1024) and image.packed_file
                checksum = hashlib.sha256(bytes(image.packed_file.data)).hexdigest()
                found.add(checksum)
                role = next(k for k, v in entry['maps'].items() if v['sha256'] == checksum)
                assert image.colorspace_settings.name == ('sRGB' if role == 'input' else 'Non-Color')
                # UI runtime dimensions/selection are not shader settings.
                for key in ['interpolation', 'projection', 'extension', 'projection_blend']:
                    assert numeric_props(node)[key] == textures[role][key]
            assert found == expected
            control = scene['review_control']
            bsdf = next(n for n in material.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
            output = next(n for n in material.node_tree.nodes if n.type == 'OUTPUT_MATERIAL')
            assert bool(bsdf.inputs['Normal'].links) == (control not in ['height', 'roughness'])
            assert bool(output.inputs['Displacement'].links) == (control not in ['normal', 'roughness'])
            assert bool(bsdf.inputs['Roughness'].links) == (control in ['combined', 'raw', 'roughness'])
        checked.append({'material': entry['id'], 'scenes_checked': 9, 'packed_maps_exact': True,
                        'template_geometry_modifiers_mapping_settings_match': True})
    # Regression for the old renderer: even if two UV layers differ, its new
    # generated surface binds color, normals and displacement to one UV basis.
    renderer_path = Path(__file__).with_name('render-material-comparison.py')
    spec = importlib.util.spec_from_file_location('shared_renderer', renderer_path)
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=2, y_subdivisions=2, size=2)
    obj = bpy.context.object
    obj.data.uv_layers.new(name='UVMap')
    material = bpy.data.materials.new('UV regression')
    material.use_nodes = True
    obj.data.materials.append(material)
    material.node_tree.nodes.new('ShaderNodeTexImage')
    material.node_tree.nodes.new('ShaderNodeNormalMap')
    modifier = obj.modifiers.new('UV displacement', 'DISPLACE')
    uv = renderer.bind_surface_uv(obj)
    assert len(obj.data.uv_layers) == 1 and uv.active_render and modifier.uv_layer == uv.name
    for node in material.node_tree.nodes:
        if node.type == 'NORMAL_MAP':
            assert node.uv_map == uv.name
        if node.type == 'TEX_IMAGE':
            assert node.inputs['Vector'].links[0].from_node.uv_map == uv.name
    result = {'at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'results': checked,
              'single_uv_regression_passed': True, 'interactive_ui_tested': False,
              'source_pixels_changed': False, 'model_assignment_changed': False}
    (folder / 'reopen-verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'files_verified': len(checked), 'scenes_verified': len(checked) * 9}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    verify(args.folder.resolve())
