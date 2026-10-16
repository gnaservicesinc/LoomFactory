"""Create packed material review scenes from a human-authored Blender template.

Blender --background --factory-startup --python this.py -- \
    --template example.blend --review artistic-review --output blender-review

The template owns topology, UVs, node mapping, material/texture/modifier
settings, Eevee and Cycles configuration. Only the four mapped images are
replaced. Native 1K files are packed verbatim. No baked/dense mesh is built.
Original template, maps, models and existing outputs are never overwritten.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys

import bpy


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def named_link(tree, node, socket):
    links = node.inputs[socket].links
    if len(links) != 1:
        raise ValueError(f'Template needs one connection into {node.name}/{socket}')
    return links[0].from_node


def mapped_nodes(material):
    tree = material.node_tree
    bsdf = next(n for n in tree.nodes if n.type == 'BSDF_PRINCIPLED')
    output = next(n for n in tree.nodes if n.type == 'OUTPUT_MATERIAL' and n.is_active_output)
    normal = named_link(tree, bsdf, 'Normal')
    displacement = named_link(tree, output, 'Displacement')
    if normal.type != 'NORMAL_MAP' or displacement.type != 'DISPLACEMENT':
        raise ValueError('Expected tangent normal and shader displacement nodes')
    result = {'input': named_link(tree, bsdf, 'Base Color'),
              'roughness': named_link(tree, bsdf, 'Roughness'),
              'normal': named_link(tree, normal, 'Color'),
              'depth': named_link(tree, displacement, 'Height')}
    if any(n.type != 'TEX_IMAGE' for n in result.values()):
        raise ValueError('Each template map must come directly from an Image Texture')
    return result, bsdf, output, normal, displacement


def disconnect(tree, node, socket):
    for link in list(node.inputs[socket].links):
        tree.links.remove(link)


def clone_scene(source, plane, name, engine, control):
    scene = source.copy()
    scene.name = name
    # A copied scene with no window user is otherwise purged on save/reopen.
    scene.use_fake_user = True
    scene.render.engine = engine
    copies = {}
    originals = list(source.objects)
    for collection in list(scene.collection.children):
        scene.collection.children.unlink(collection)
    for original in list(scene.collection.objects):
        scene.collection.objects.unlink(original)
    for original in originals:
        obj = original.copy()
        if original.type in ['MESH', 'CAMERA']:
            obj.data = original.data.copy()
        scene.collection.objects.link(obj)
        copies[original.name] = obj
    surface = copies[plane.name]
    material = plane.active_material.copy()
    surface.data.materials.clear()
    surface.data.materials.append(material)
    if source.camera:
        scene.camera = copies[source.camera.name]
    nodes, bsdf, output, normal, displacement = mapped_nodes(material)
    tree = material.node_tree
    if control == 'raw':
        scene.cycles.use_denoising = False
        scene.cycles.use_preview_denoising = False
    elif control in ['normal', 'height', 'roughness']:
        # Neutral controls isolate lighting contributions, not map similarity.
        disconnect(tree, bsdf, 'Base Color')
        bsdf.inputs['Base Color'].default_value = (.5, .5, .5, 1)
        if control != 'roughness':
            disconnect(tree, bsdf, 'Roughness')
            bsdf.inputs['Roughness'].default_value = .85
        if control != 'normal':
            disconnect(tree, bsdf, 'Normal')
        if control != 'height':
            disconnect(tree, output, 'Displacement')
    elif control == 'matte':
        disconnect(tree, bsdf, 'Roughness')
        bsdf.inputs['Roughness'].default_value = .85
    scene['review_control'] = control
    return scene


def build(template, review, output, only=None):
    before = sha(template)
    config = json.loads((review / 'viewer-configuration.json').read_text())
    output.mkdir(parents=True, exist_ok=True)
    saved_template = output / 'User-example-original.blend'
    if saved_template.exists():
        if sha(saved_template) != before:
            raise FileExistsError(saved_template)
    else:
        shutil.copyfile(template, saved_template)
    results = []
    for entry in config['materials']:
        if only and entry['id'] not in only:
            continue
        dest = output / (entry['id'] + '.blend')
        if dest.exists():
            raise FileExistsError(dest)
        bpy.ops.wm.open_mainfile(filepath=str(template))
        source = bpy.context.scene
        planes = [o for o in source.objects if o.type == 'MESH']
        if len(planes) != 1:
            raise ValueError('Use a single-surface example template')
        plane = planes[0]
        if len(plane.data.vertices) != 4 or len(plane.data.polygons) != 1:
            raise ValueError('Keep the simple quad template; do not substitute a baked mesh')
        if len(plane.data.uv_layers) != 1:
            raise ValueError('Template must have a single unambiguous UV map')
        uv = plane.data.uv_layers[0]
        uv.active_render = True
        material = plane.active_material
        nodes, bsdf, material_out, normal, displacement = mapped_nodes(material)
        normal.uv_map = uv.name
        numeric_settings = {'normal_strength': normal.inputs['Strength'].default_value,
                            'normal_base': normal.base, 'normal_convention': normal.convention,
                            'displacement_scale': displacement.inputs['Scale'].default_value,
                            'midpoint': displacement.inputs['Midlevel'].default_value,
                            'displacement_method': material.displacement_method}
        maps = {}
        for role, node in nodes.items():
            path = (review / entry['maps'][role]).resolve()
            source_sha = sha(path)
            image = bpy.data.images.load(str(path), check_existing=False)
            image.colorspace_settings.name = 'sRGB' if role == 'input' else 'Non-Color'
            if tuple(image.size) != (1024, 1024):
                raise ValueError(f'{role}: use an existing native 1K map; no resize')
            image.pack()
            if hashlib.sha256(bytes(image.packed_file.data)).hexdigest() != source_sha:
                raise ValueError('Packed image bytes differ from the original')
            # Portable names for a later manual unpack; packed bytes stay exact.
            image.filepath = '//textures/' + entry['id'] + '-' + role + path.suffix
            node.image = image
            maps[role] = {'path': str(path), 'sha256': source_sha,
                          'packed_sha256': source_sha, 'size': list(image.size),
                          'color_space': image.colorspace_settings.name,
                          'interpolation': node.interpolation, 'extension': node.extension}
        for node in source.world.node_tree.nodes:
            if node.type == 'BACKGROUND':
                node.inputs['Strength'].default_value = .5
        source.render.threads_mode = 'AUTO'
        source.cycles.pixel_filter_type = 'BLACKMAN_HARRIS'
        source.cycles.filter_width = 1
        source.render.filepath = '//renders/' + entry['id'] + '-cycles.png'
        source.name = '01 Cycles - Combined'
        source.use_fake_user = True
        source['review_control'] = 'combined'
        source['review_material'] = entry['id']
        source['review_runs'] = json.dumps(entry['runs'])
        # Exact Eevee template settings remain stored even when Cycles is active.
        eevee = clone_scene(source, plane, '02 Eevee - Combined', 'CYCLES', 'combined')
        eevee_engine = 'CYCLES'  # Resolve registered engines through assignment.
        for candidate in ['BLENDER_EEVEE_NEXT', 'BLENDER_EEVEE']:
            try:
                eevee.render.engine = candidate
                eevee_engine = candidate
                break
            except TypeError:
                continue
        if eevee_engine == 'CYCLES':
            raise RuntimeError('Eevee is unavailable in this Blender build')
        definitions = [('03 Cycles - No denoising', 'CYCLES', 'raw'),
                       ('04 Eevee - Normal only', eevee_engine, 'normal'),
                       ('05 Eevee - Height only', eevee_engine, 'height'),
                       ('06 Eevee - Roughness only', eevee_engine, 'roughness'),
                       ('07 Eevee - Matte control', eevee_engine, 'matte'),
                       ('08 Cycles - Height only', 'CYCLES', 'height'),
                       ('09 Cycles - Matte control', 'CYCLES', 'matte')]
        for name, engine, control in definitions:
            clone_scene(source, plane, name, engine, control)
        # Keep rendered view tied to the actual .5 world, not a separate
        # material-preview studio brightness. Preserve the user's camera view.
        for screen in bpy.data.screens:
            for area in screen.areas:
                if area.type == 'VIEW_3D':
                    area.spaces.active.shading.use_scene_world = True
                    area.spaces.active.shading.use_scene_world_render = True
                    area.spaces.active.shading.use_scene_lights = True
                    area.spaces.active.shading.use_scene_lights_render = True
        for image in list(bpy.data.images):
            if image.users == 0:
                bpy.data.images.remove(image)
        if bpy.context.window:
            bpy.context.window.scene = source
        text = bpy.data.texts.new('START HERE - material review')
        text.write(f'''{entry['name']}
Generated model IDs: {json.dumps(entry['runs'], indent=2)}

Based directly on the user's saved example; original template preserved.
Native 1K maps packed byte-for-byte. No dense object/OBJ is generated.
Shared UV -> Mapping drives color, normal, roughness and height textures.
Tangent OpenGL normal, Displaced Base. Shader displacement .025, midpoint .5.
Simple quad plus the original Simple adaptive subdivision modifier.
World strength .5. Cycles Blackman-Harris pixel width1. CPU Auto-Detect.

Scenes: 01 Cycles Combined; 02 Eevee Combined; 03 Cycles No denoising;
04 Eevee Normal only; 05 Eevee Height only; 06 Eevee Roughness only;
07 Eevee Matte control; 08 Cycles Height only; 09 Cycles Matte control.
Matching scenes retain the template denoising; scene03 disables it explicitly.
Eevee is a fast shading diagnostic, not Cycles adaptive geometry parity.
The matte control uses roughness .85 without editing the prediction.

Rotate the view to inspect. Rotate the world Environment Texture mapping Z
to move illumination. The original adaptive Object edge length .0001 and
max subdivision12 are retained; Cycles can be expensive at fine tessellation.
These are artistic comparisons, never automatic model approval/selection.
''')
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=str(dest))
        assert sha(template) == before
        for record in maps.values():
            assert sha(record['path']) == record['sha256']
        results.append({'id': entry['id'], 'name': entry['name'], 'file': str(dest),
                        'sha256': sha(dest), 'runs': entry['runs'], 'maps': maps,
                        'material_settings': numeric_settings, 'scenes': [s.name for s in bpy.data.scenes],
                        'world_strength': .5, 'pixel_filter': 'BLACKMAN_HARRIS',
                        'pixel_filter_width': source.cycles.filter_width,
                        'threads_mode': source.render.threads_mode,
                        'detected_threads': source.render.threads,
                        'template_sha256_unchanged': before})
    receipt = {'at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'template': str(saved_template), 'template_sha256': before,
               'builder_sha256': sha(__file__), 'results': results,
               'source_pixels_changed': False, 'model_assignment_changed': False}
    (output / 'build-receipt.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps({'files': len(results), 'receipt': str(output / 'build-receipt.json')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--template', required=True, type=Path)
    p.add_argument('--review', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--only', nargs='*')
    args = p.parse_args(sys.argv[sys.argv.index('--') + 1:])
    build(args.template.resolve(), args.review.resolve(), args.output.resolve(), args.only)
