"""Blender CPU review of an aligned depth/normal/roughness candidate set.

Uses original numeric maps unchanged. Fixed geometry, displacement scale,
camera, color management and lights apply to both source and candidate renders.
This is a review artifact, not a replacement or transformation of saved maps.
"""
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector


def look_at(obj, point):
    obj.rotation_euler = (Vector(point) - obj.location).to_track_quat('-Z', 'Y').to_euler()


def load_image(path, numeric=False):
    image = bpy.data.images.load(str(path), check_existing=False)
    image.colorspace_settings.name = 'Non-Color' if numeric else 'sRGB'
    return image


def render_set(record, prefix, output, lighting, neutral=False):
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 24
    scene.cycles.use_denoising = False
    scene.render.threads_mode = 'FIXED'
    scene.render.threads = 4
    scene.render.resolution_x = 768
    scene.render.resolution_y = 640
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.view_settings.view_transform = 'AgX'
    scene.view_settings.exposure = 0
    scene.world.color = (0.08, 0.08, 0.08)

    material = bpy.data.materials.new('Matched three-map material')
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    bsdf = nodes.get('Principled BSDF')
    bsdf.inputs['Metallic'].default_value = 0
    bsdf.inputs['Base Color'].default_value = (0.5, 0.5, 0.5, 1)
    if not neutral:
        diffuse = nodes.new('ShaderNodeTexImage')
        diffuse.image = load_image(record['input'])
        links.new(diffuse.outputs['Color'], bsdf.inputs['Base Color'])
    roughness = nodes.new('ShaderNodeTexImage')
    roughness.image = load_image(record[prefix + 'roughness'], True)
    links.new(roughness.outputs['Color'], bsdf.inputs['Roughness'])
    normal_image = nodes.new('ShaderNodeTexImage')
    normal_image.image = load_image(record[prefix + 'normal'], True)
    normal = nodes.new('ShaderNodeNormalMap')
    normal.space = 'TANGENT'
    normal.inputs['Strength'].default_value = 1
    links.new(normal_image.outputs['Color'], normal.inputs['Color'])
    links.new(normal.outputs['Normal'], bsdf.inputs['Normal'])

    # The same dense grid and physical height scale are used for both versions.
    # No per-image normalization or contrast adjustment is applied to the maps.
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=192, y_subdivisions=192, size=2)
    plane = bpy.context.object
    plane.data.materials.append(material)
    uv = plane.data.uv_layers.new(name='UVMap')
    for polygon in plane.data.polygons:
        polygon.use_smooth = True
        for loop_index in polygon.loop_indices:
            vertex = plane.data.vertices[plane.data.loops[loop_index].vertex_index]
            uv.data[loop_index].uv = ((vertex.co.x + 1) / 2, (vertex.co.y + 1) / 2)
    height = bpy.data.textures.new('Raw native height', type='IMAGE')
    height.image = load_image(record[prefix + 'depth'], True)
    displacement = plane.modifiers.new('Fixed physical relief', 'DISPLACE')
    displacement.texture = height
    displacement.texture_coords = 'UV'
    displacement.uv_layer = 'UVMap'
    displacement.mid_level = 0.5
    displacement.strength = 0.08

    bpy.ops.object.camera_add(location=(1.4, -2.2, 2.6))
    camera = bpy.context.object
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = 3.2
    look_at(camera, (0, 0, 0))
    scene.camera = camera
    position = (-2.5, -1.2, 1.3) if lighting == 'left' else (2.5, 0.8, 1.3)
    bpy.ops.object.light_add(type='AREA', location=position)
    key = bpy.context.object
    key.data.energy = 350
    key.data.shape = 'DISK'
    key.data.size = 2
    look_at(key, (0, 0, 0))
    bpy.ops.object.light_add(type='AREA', location=(0, -0.5, 4))
    fill = bpy.context.object
    fill.data.energy = 60
    fill.data.size = 4
    look_at(fill, (0, 0, 0))
    scene.render.filepath = str(output)
    bpy.ops.render.render(write_still=True)


path = Path(sys.argv[sys.argv.index('--') + 1])
record = json.loads(path.read_text())
if not all(record.get(prefix + role) for prefix in ['', 'original_'] for role in ['depth', 'normal', 'roughness']):
    raise RuntimeError('A complete source and candidate map set is required for joint review')
directory = path.parent / (path.stem + '-renders')
directory.mkdir(exist_ok=True)
for lighting in ['left', 'right']:
    for label, prefix in [('source', 'original_'), ('candidate', '')]:
        for neutral in [False, True]:
            suffix = '-neutral' if neutral else ''
            render_set(record, prefix, directory / f'{label}-{lighting}{suffix}.png', lighting, neutral)
receipt = {
    'schema': 'loom-three-map-render-review-v1',
    'input': record['input'], 'maps': record, 'render_directory': str(directory),
    'lights': ['left', 'right'], 'neutral_and_diffuse_views': True,
    'displacement_strength': 0.08, 'displacement_midlevel': 0.5,
    'normal_convention': 'OpenGL +Y', 'roughness_color_space': 'Non-Color',
    'note': 'Fixed artistic review scale, not measured physical relief. Original map files are never rewritten.'
}
(directory / 'render-receipt.json').write_text(json.dumps(receipt, indent=2))
