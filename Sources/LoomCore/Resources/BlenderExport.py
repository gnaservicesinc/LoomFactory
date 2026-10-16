"""Loom Factory native-map review. Run with Blender --background --factory-startup
--python BlenderExport.py -- /path/to/review.json. Saves, never renders or trains.
"""
import hashlib
import json
import math
from pathlib import Path
import sys

import bpy

IMAGES = {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image(root, record, role):
    path = (root / record['file']).resolve()
    if not path.is_relative_to(root.resolve()) or sha(path) != record['sha256']:
        raise ValueError('Image path or checksum changed: ' + str(path))
    key = (str(path), role)
    if key in IMAGES:
        return IMAGES[key]
    img = bpy.data.images.load(str(path), check_existing=False)
    img.colorspace_settings.name = 'sRGB' if role == 'color' else 'Non-Color'
    img.pack()
    if hashlib.sha256(bytes(img.packed_file.data)).hexdigest() != record['sha256']:
        raise ValueError('Packed image bytes changed')
    img.filepath = '//' + record['file']
    IMAGES[key] = img
    return img


def surface(root, config, maps, label, x, control):
    aspect = config['height'] / config['width']
    # A quad with unapplied shader displacement, never a million-vertex OBJ.
    mesh = bpy.data.meshes.new(label)
    mesh.from_pydata([(-1, -aspect, 0), (1, -aspect, 0),
                      (1, aspect, 0), (-1, aspect, 0)], [], [(0, 1, 2, 3)])
    mesh.uv_layers.new(name='UVMap')
    for loop, uv in zip(mesh.uv_layers.active.data, [(0, 0), (1, 0), (1, 1), (0, 1)]):
        loop.uv = uv
    mesh.uv_layers.active.active_render = True
    obj = bpy.data.objects.new(label, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.location.x = x
    obj['model_names'] = label
    obj['exact_map_versions'] = json.dumps(maps, sort_keys=True)
    sub = obj.modifiers.new('Simple adaptive subdivision', 'SUBSURF')
    sub.subdivision_type = 'SIMPLE'
    sub.levels = sub.render_levels = 6
    sub.quality = 6
    # Blender 5.4 moved adaptive settings from the object to the modifier.
    if hasattr(sub, 'use_adaptive_subdivision'):
        sub.use_adaptive_subdivision = True
        sub.adaptive_space = 'OBJECT'
        sub.adaptive_object_edge_length = .0001
        sub.adaptive_pixel_size = 1
    elif hasattr(obj, 'cycles') and hasattr(obj.cycles, 'use_adaptive_subdivision'):
        obj.cycles.use_adaptive_subdivision = True
    mat = bpy.data.materials.new(label)
    mesh.materials.append(mat)
    mat.use_nodes = True
    if hasattr(mat, 'displacement_method'):
        mat.displacement_method = 'BOTH'
    elif hasattr(mat, 'cycles'):
        mat.cycles.displacement_method = 'BOTH'
    mat['model_names'] = label
    mat['exact_map_versions'] = json.dumps(maps, sort_keys=True)
    tree = mat.node_tree
    tree.nodes.clear()
    def node(kind, name, xy):
        n = tree.nodes.new(kind)
        n.name = n.label = name
        n.location = xy
        return n
    coord = node('ShaderNodeTexCoord', 'One UVMap', (-1100, 200))
    mapping = node('ShaderNodeMapping', 'Shared map coordinates', (-900, 200))
    tree.links.new(coord.outputs['UV'], mapping.inputs['Vector'])
    bsdf = node('ShaderNodeBsdfPrincipled', label, (200, 200))
    bsdf.inputs['Roughness'].default_value = .85
    bsdf.inputs['Base Color'].default_value = (.5, .5, .5, 1)
    output = node('ShaderNodeOutputMaterial', 'Material output', (550, 200))
    tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    records = {'color': config['color'], **maps}
    for index, (role, record) in enumerate(records.items()):
        texture = node('ShaderNodeTexImage', role + ' · ' + record.get('model_name', 'Source'), (-580, 450-index*300))
        texture.image = image(root, record, role)
        if list(texture.image.size) != [config['width'], config['height']]:
            raise ValueError('Map size differs from native color input')
        texture.interpolation = 'Linear'
        texture.extension = 'REPEAT'
        tree.links.new(mapping.outputs['Vector'], texture.inputs['Vector'])
        if role == 'color' and control not in ['height', 'normal', 'roughness']:
            tree.links.new(texture.outputs['Color'], bsdf.inputs['Base Color'])
        elif role == 'roughness' and control in ['combined', 'raw', 'roughness']:
            tree.links.new(texture.outputs['Color'], bsdf.inputs['Roughness'])
        elif role == 'normal' and control in ['combined', 'raw', 'normal', 'matte']:
            normal = node('ShaderNodeNormalMap', 'Tangent normal · ' + record.get('model_name', 'Source'), (-100, -150))
            normal.space = 'TANGENT'
            normal.uv_map = 'UVMap'
            if hasattr(normal, 'base'):
                normal.base = 'DISPLACED'
            convention = record.get('normal_convention', '').lower()
            directx = 'directx' in convention or convention == 'dx'
            if hasattr(normal, 'convention'):
                normal.convention = 'DIRECTX' if directx else 'OPENGL'
            elif directx:
                raise ValueError('This Blender version cannot explicitly decode DirectX source normals')
            tree.links.new(texture.outputs['Color'], normal.inputs['Color'])
            tree.links.new(normal.outputs['Normal'], bsdf.inputs['Normal'])
        elif role == 'depth' and control in ['combined', 'raw', 'height', 'matte']:
            displacement = node('ShaderNodeDisplacement', 'Height · ' + record.get('model_name', 'Source'), (200, -400))
            displacement.space = 'OBJECT'
            displacement.inputs['Scale'].default_value = config['displacement_scale']
            displacement.inputs['Midlevel'].default_value = config['midpoint']
            tree.links.new(texture.outputs['Color'], displacement.inputs['Height'])
            tree.links.new(displacement.outputs['Displacement'], output.inputs['Displacement'])
    return obj


def world():
    w = bpy.data.worlds.new('Environment · strength 0.5 · rotate Mapping Z')
    w.use_nodes = True
    tree = w.node_tree
    bg = next(n for n in tree.nodes if n.type == 'BACKGROUND')
    bg.inputs['Strength'].default_value = .5
    folder = Path(bpy.utils.system_resource('DATAFILES', path='studiolights/world'))
    choices = sorted(folder.glob('*.exr'))
    environment = next((p for p in choices if p.stem == 'studio'), choices[0] if choices else None)
    if environment:
        tex = tree.nodes.new('ShaderNodeTexEnvironment')
        tex.image = bpy.data.images.load(str(environment))
        tex.image.pack()
        tex.image.filepath = '//environment.exr'
        coord = tree.nodes.new('ShaderNodeTexCoord')
        mapping = tree.nodes.new('ShaderNodeMapping')
        mapping.name = mapping.label = 'Rotate environment Z to move lighting'
        tree.links.new(coord.outputs['Normal'], mapping.inputs['Vector'])
        tree.links.new(mapping.outputs['Vector'], tex.inputs['Vector'])
        tree.links.new(tex.outputs['Color'], bg.inputs['Color'])
    return w


def build(manifest):
    root = manifest.parent
    config = json.loads(manifest.read_text())
    if config['schema'] != 'loom-blender-review-v1':
        raise ValueError('Unknown Blender review schema')
    dest = root / 'Material Review.blend'
    if dest.exists():
        raise FileExistsError(dest)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    lighting = world()
    names = [r['model_name'] for r in config['generated'].values()]
    identity = ' | '.join(names)
    definitions = [('01 Cycles - Combined', 'CYCLES', 'combined'),
                   ('02 Eevee - Combined', 'BLENDER_EEVEE_NEXT', 'combined'),
                   ('03 Cycles - No denoising', 'CYCLES', 'raw'),
                   ('04 Eevee - Normal only', 'BLENDER_EEVEE_NEXT', 'normal'),
                   ('05 Eevee - Height only', 'BLENDER_EEVEE_NEXT', 'height'),
                   ('06 Eevee - Roughness only', 'BLENDER_EEVEE_NEXT', 'roughness'),
                   ('07 Eevee - Matte control', 'BLENDER_EEVEE_NEXT', 'matte'),
                   ('08 Cycles - Height only', 'CYCLES', 'height'),
                   ('09 Cycles - Matte control', 'CYCLES', 'matte')]
    for i, (name, engine, control) in enumerate(definitions):
        scene = bpy.context.scene if i == 0 else bpy.data.scenes.new(name)
        bpy.context.window.scene = scene
        scene.name = name
        scene.use_fake_user = True
        try:
            scene.render.engine = engine
        except TypeError:
            scene.render.engine = 'BLENDER_EEVEE'
        scene.world = lighting
        scene.render.threads_mode = 'AUTO'
        scene.render.resolution_x = scene.render.resolution_y = 1024
        scene.render.resolution_percentage = 100
        scene.render.film_transparent = True
        scene.cycles.device = 'CPU'
        scene.cycles.samples = 96
        scene.cycles.preview_samples = 1024
        scene.cycles.pixel_filter_type = 'BLACKMAN_HARRIS'
        scene.cycles.filter_width = 1
        scene.cycles.use_denoising = control != 'raw'
        scene.cycles.use_preview_denoising = control != 'raw'
        scene.cycles.dicing_rate = 1
        scene.cycles.preview_dicing_rate = 8
        scene.cycles.max_subdivisions = 12
        scene.view_settings.view_transform = 'AgX'
        scene.view_settings.look = 'None'
        for key, value in [('use_pixel_jitter', False), ('use_adaptive_sampling', True),
                           ('adaptive_threshold', .01), ('adaptive_min_samples', 1),
                           ('use_preview_adaptive_sampling', True), ('preview_adaptive_threshold', .1),
                           ('preview_adaptive_min_samples', 1), ('offscreen_dicing_scale', 4),
                           ('use_guiding', True), ('use_deterministic_guiding', True)]:
            if hasattr(scene.cycles, key):
                setattr(scene.cycles, key, value)
        if hasattr(scene, 'eevee'):
            eevee = scene.eevee
            for key, value in [('taa_render_samples', 128), ('taa_samples', 16),
                               ('use_raytracing', True), ('ray_tracing_method', 'SCREEN'),
                               ('use_fast_gi', True), ('fast_gi_method', 'GLOBAL_ILLUMINATION'),
                               ('fast_gi_resolution', '1'), ('fast_gi_ray_count', 16), ('fast_gi_step_count', 64),
                               ('fast_gi_quality', .25), ('fast_gi_thickness_near', .1), ('fast_gi_distance', 0),
                               ('fast_gi_bias', .05), ('use_shadows', True), ('shadow_ray_count', 4),
                               ('shadow_step_count', 16), ('shadow_resolution_scale', 1)]:
                if hasattr(eevee, key):
                    setattr(eevee, key, value)
            if hasattr(eevee, 'ray_tracing_options'):
                eevee.ray_tracing_options.resolution_scale = '1'
                eevee.ray_tracing_options.use_denoise = True
        compare = config['include_source']
        generated = surface(root, config, config['generated'], identity, -1.15 if compare else 0, control)
        if compare:
            surface(root, config, config['source'], 'Source material · ' + config['source_name'], 1.15, control)
        camera = bpy.data.objects.new('Review camera', bpy.data.cameras.new('Review camera'))
        scene.collection.objects.link(camera)
        camera.location = (0, -4.8 if compare else -3.2, 4.2 if compare else 2.8)
        camera.rotation_euler = (-camera.location).to_track_quat('-Z', 'Y').to_euler()
        scene.camera = camera
        bpy.context.view_layer.objects.active = generated
        generated.select_set(True)
        scene['loom_review_manifest'] = json.dumps(config, sort_keys=True)
    bpy.context.window.scene = bpy.data.scenes['01 Cycles - Combined']
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                area.spaces.active.shading.use_scene_world = True
                area.spaces.active.shading.use_scene_world_render = True
                area.spaces.active.region_3d.view_distance = 6 if config['include_source'] else 4
    text = bpy.data.texts.new('START HERE - exact tested models')
    text.write('Generated object/material: ' + identity + '\n\n' + json.dumps(config, indent=2) +
               '\n\nNative image bytes packed. One UVMap per quad. Missing height/normal: flat; missing roughness: .85. '
               'World .5, shader displacement .025 / midpoint .5, independent normal strength. '
               'Switch scenes to isolate maps. Eevee is a shading diagnostic; Cycles provides adaptive displacement. '
               'Preview dicing 8 is coarser than final dicing 1. CPU Auto-Detect; pixel filter width 1. '
               'Export does not render, train, approve or select models.\n')
    bpy.ops.wm.save_as_mainfile(filepath=str(dest))
    (root / 'export-receipt.json').write_text(json.dumps({
        'blend': dest.name, 'sha256': sha(dest), 'scenes': [s.name for s in bpy.data.scenes],
        'generated_models': names, 'source_plane': config['include_source'],
        'image_pixels_modified': False, 'rendered': False, 'blender_version': bpy.app.version_string}, indent=2))


if __name__ == '__main__':
    build(Path(sys.argv[sys.argv.index('--') + 1]).resolve())
