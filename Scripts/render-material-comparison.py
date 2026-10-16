"""Blender CPU review of an aligned depth/normal/roughness candidate set.

Uses original numeric maps unchanged. Fixed geometry, displacement scale,
camera, color management and lights apply to both source and candidate renders.
This is a review artifact, not a replacement or transformation of saved maps.
"""
import json
import math
from pathlib import Path
import shutil
import sys

import bpy
from mathutils import Vector


def look_at(obj, point):
    obj.rotation_euler = (Vector(point) - obj.location).to_track_quat('-Z', 'Y').to_euler()


def load_image(path, numeric=False):
    image = bpy.data.images.load(str(path), check_existing=True)
    image.colorspace_settings.name = 'Non-Color' if numeric else 'sRGB'
    return image


def review_grid_cells(width, height, pixels_per_node, max_edge_vertices):
    """Bound preview geometry independently of native map resolution."""
    if min(width, height) < 2 or pixels_per_node < 1 or max_edge_vertices < 2:
        raise ValueError('Review mesh dimensions, spacing and vertex cap must be positive')
    columns = math.ceil((width - 1) / pixels_per_node)
    rows = math.ceil((height - 1) / pixels_per_node)
    scale = min(1, (max_edge_vertices - 1) / max(columns, rows))
    return max(1, math.floor(columns * scale)), max(1, math.floor(rows * scale))


def bind_surface_uv(plane):
    """Use one explicit UV basis for displacement and every material texture.

    primitive_grid_add already creates a UV layer. Creating another UVMap
    silently receives a .001 suffix, so relying on active/default names can
    make image sampling and tangent-space normals use different bases.
    This operates only on the newly generated review mesh.
    """
    while plane.data.uv_layers:
        plane.data.uv_layers.remove(plane.data.uv_layers[0])
    uv = plane.data.uv_layers.new(name='MaterialUV')
    uv.active_render = True
    plane.data.uv_layers.active = uv
    for polygon in plane.data.polygons:
        polygon.use_smooth = True
        for loop_index in polygon.loop_indices:
            vertex = plane.data.vertices[plane.data.loops[loop_index].vertex_index]
            uv.data[loop_index].uv = ((vertex.co.x + 1) / 2, (vertex.co.y + 1) / 2)
    for material in plane.data.materials:
        tree = material.node_tree
        uv_node = tree.nodes.new('ShaderNodeUVMap')
        uv_node.name = 'Explicit material UV'
        uv_node.uv_map = uv.name
        for node in tree.nodes:
            if node.type == 'TEX_IMAGE':
                tree.links.new(uv_node.outputs['UV'], node.inputs['Vector'])
            elif node.type == 'NORMAL_MAP':
                node.uv_map = uv.name
    for modifier in plane.modifiers:
        if modifier.type == 'DISPLACE':
            modifier.texture_coords = 'UV'
            modifier.uv_layer = uv.name
    return uv


def export_textured_obj(plane, path, diffuse):
    """Bake actual displacement geometry; OBJ/MTL carries the diffuse overlay.

    The editable .blend preserves the full node material. OBJ is intentionally
    a geometry-plus-color inspection asset, independent of normal/rough maps.
    """
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    if path.exists():
        raise FileExistsError(path)
    texture = path.parent / ('diffuse' + Path(diffuse).suffix)
    if not texture.exists():
        shutil.copy2(diffuse, texture)
    mtl = path.with_suffix('.mtl')
    mtl.write_text('newmtl surface\nKd 1 1 1\nKs 0.04 0.04 0.04\nNs 20\nillum 2\nmap_Kd ' + texture.name + '\n')
    evaluated = plane.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        # Displace changes positions without changing the source UV topology.
        uv_by_vertex = {}
        for loop in mesh.loops:
            uv_by_vertex[loop.vertex_index] = mesh.uv_layers.active.data[loop.index].uv.copy()
        with path.open('x') as f:
            f.write('# Baked displacement and original diffuse; artistic review\nmtllib ' + mtl.name + '\no material_surface\n')
            for v in mesh.vertices:
                f.write(f'v {v.co.x:.9g} {v.co.y:.9g} {v.co.z:.9g}\n')
            for i in range(len(mesh.vertices)):
                uv = uv_by_vertex[i]
                f.write(f'vt {uv.x:.9g} {uv.y:.9g}\n')
            for v in mesh.vertices:
                f.write(f'vn {v.normal.x:.9g} {v.normal.y:.9g} {v.normal.z:.9g}\n')
            f.write('usemtl surface\ns 1\n')
            for polygon in mesh.polygons:
                f.write('f ' + ' '.join(f'{i+1}/{i+1}/{i+1}' for i in polygon.vertices) + '\n')
        return {'path': str(path), 'vertices': len(mesh.vertices), 'faces': len(mesh.polygons),
                'z_range': [min(v.co.z for v in mesh.vertices), max(v.co.z for v in mesh.vertices)],
                'diffuse': str(texture), 'normal_roughness': 'Not embedded in OBJ material; inspect full material in .blend.'}
    finally:
        evaluated.to_mesh_clear()


def render_set(record, prefix, output, lighting, neutral=False, options=None):
    options = options or {}
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 24
    scene.cycles.use_denoising = False
    scene.render.threads_mode = 'AUTO'
    scene.cycles.pixel_filter_type = 'BLACKMAN_HARRIS'
    scene.cycles.filter_width = 1
    scene.render.resolution_x = 768
    scene.render.resolution_y = 640
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.view_settings.view_transform = 'AgX'
    scene.view_settings.exposure = 0
    scene.world.color = (0.08, 0.08, 0.08)
    if options.get('environment'):
        scene.world.use_nodes = True
        world_nodes = scene.world.node_tree.nodes
        world_nodes.clear()
        environment = world_nodes.new('ShaderNodeTexEnvironment')
        environment.image = load_image(options['environment'], True)
        # Leave Vector unconnected so Cycles supplies its world ray direction.
        # A surface-normal coordinate link is not valid environment lighting.
        environment.texture_mapping.rotation[2] = math.radians(options.get('environment_rotation', 0))
        background = world_nodes.new('ShaderNodeBackground')
        background.inputs['Strength'].default_value = options.get('environment_strength', 0.05)
        world_out = world_nodes.new('ShaderNodeOutputWorld')
        world_links = scene.world.node_tree.links
        world_links.new(environment.outputs['Color'], background.inputs['Color'])
        world_links.new(background.outputs['Background'], world_out.inputs['Surface'])

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
    if options.get('roughness_constant') is None:
        roughness = nodes.new('ShaderNodeTexImage')
        roughness.image = load_image(record[prefix + 'roughness'], True)
        links.new(roughness.outputs['Color'], bsdf.inputs['Roughness'])
    else:
        bsdf.inputs['Roughness'].default_value = options['roughness_constant']
    normal_image = nodes.new('ShaderNodeTexImage')
    normal_image.image = load_image(record[prefix + 'normal'], True)
    normal = nodes.new('ShaderNodeNormalMap')
    normal.space = 'TANGENT'
    normal.name = 'Artistic normal strength'
    normal.inputs['Strength'].default_value = options.get('normal_strength', 1)
    links.new(normal_image.outputs['Color'], normal.inputs['Color'])
    if options.get('use_normal', True):
        links.new(normal.outputs['Normal'], bsdf.inputs['Normal'])

    # Legacy reviews retain their original settings. Artistic reviews expose
    # mesh density and independent gains; these never rewrite numeric maps.
    height_image = load_image(record[prefix + 'depth'], True)
    columns = rows = 192
    if options.get('pixels_per_node'):
        columns, rows = review_grid_cells(height_image.size[0], height_image.size[1],
                                         options['pixels_per_node'],
                                         options.get('max_mesh_edge_vertices', 513))
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=columns, y_subdivisions=rows, size=2)
    plane = bpy.context.object
    plane.data.materials.append(material)
    height = bpy.data.textures.new('Raw native height', type='IMAGE')
    height.image = height_image
    displacement = plane.modifiers.new('Fixed physical relief', 'DISPLACE')
    displacement.texture = height
    displacement.texture_coords = 'UV'
    bind_surface_uv(plane)
    displacement.mid_level = options.get('height_midpoint', 0.5)
    displacement.strength = 0.08 * options.get('height_gain', 1)
    exported = None
    if options.get('export_obj'):
        exported = export_textured_obj(plane, options['export_obj'], record['input'])
    if options.get('export_only'):
        return exported

    bpy.ops.object.camera_add(location=(1.4, -2.2, 2.6))
    camera = bpy.context.object
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = 3.2
    look_at(camera, (0, 0, 0))
    scene.camera = camera
    if not options.get('environment') or lighting == 'grazing':
        position = (-2.5, -1.2, 0.45 if lighting == 'grazing' else 1.3) if lighting != 'right' else (2.5, 0.8, 1.3)
        bpy.ops.object.light_add(type='AREA', location=position)
        key = bpy.context.object
        key.data.energy = 350
        key.data.shape = 'DISK'
        key.data.size = 0.8 if lighting == 'grazing' else 2
        look_at(key, (0, 0, 0))
        if not options.get('environment'):
            bpy.ops.object.light_add(type='AREA', location=(0, -0.5, 4))
            fill = bpy.context.object
            fill.data.energy = 60
            fill.data.size = 4
            look_at(fill, (0, 0, 0))
    plane.name = 'Displaced surface — independent height, normal and roughness controls'
    plane['height_gain'] = options.get('height_gain', 1)
    plane['normal_strength'] = options.get('normal_strength', 1)
    plane['review_note'] = 'Artistic appearance review. Edit displacement modifier and Normal Map node independently. No reference-match score.'
    scene.render.filepath = str(output)
    if options.get('save_blend'):
        path = Path(options['save_blend'])
        if path.exists():
            raise FileExistsError(path)
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=str(path))
    bpy.ops.render.render(write_still=True)
    return exported


def main():
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


if __name__ == '__main__':
    main()
