"""Read-only numeric review of matched maps. Requires NumPy, Pillow and ffmpeg.

PNG scalar/RGB values decode at 16 bits without display gamma. Candidate EXR
is the native uncompressed Float32 format. Metrics do not change any map.
"""
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import numpy as np
from PIL import Image


def read_exr(path):
    data = path.read_bytes()
    if struct.unpack_from('<I', data)[0] != 20000630:
        raise ValueError('Not an OpenEXR file')
    pos, attrs = 8, {}
    while data[pos]:
        end = data.index(0, pos); name = data[pos:end].decode(); pos = end + 1
        pos = data.index(0, pos) + 1
        size = struct.unpack_from('<I', data, pos)[0]; pos += 4
        attrs[name] = data[pos:pos + size]; pos += size
    pos += 1
    if attrs['compression'] != bytes([0]):
        raise ValueError('Numeric review requires uncompressed native Float32 EXR')
    names, channel_pos = [], 0
    while attrs['channels'][channel_pos]:
        end = attrs['channels'].index(0, channel_pos)
        name = attrs['channels'][channel_pos:end].decode()
        kind, = struct.unpack_from('<i', attrs['channels'], end + 1)
        if kind != 2: raise ValueError('Expected Float32 EXR channel')
        names.append(name); channel_pos = end + 17
    x0, y0, x1, y1 = struct.unpack('<iiii', attrs['dataWindow'])
    width, height = x1-x0+1, y1-y0+1
    values = np.empty((height, len(names), width), np.float32)
    for row in range(height):
        offset, = struct.unpack_from('<Q', data, pos + row * 8)
        y, count = struct.unpack_from('<ii', data, offset)
        if count != width * len(names) * 4: raise ValueError('Unexpected EXR row')
        values[y-y0] = np.frombuffer(data, '<f4', width*len(names), offset+8).reshape(len(names), width)
    if names == ['Y']: return values[:,0,:,None]
    return values[:,[names.index(c) for c in 'RGB'],:].transpose(0,2,1)


def read_numeric(path, role):
    if path.suffix.lower() == '.exr': return read_exr(path).astype(np.float64)
    with Image.open(path) as image: width, height = image.size
    decoder = shutil.which('ffmpeg')
    if not decoder: raise RuntimeError('ffmpeg is required for exact 16-bit RGB PNG decoding')
    raw = subprocess.check_output([decoder, '-v', 'error', '-i', str(path), '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb48le', 'pipe:1'])
    rgb = np.frombuffer(raw, '<u2').reshape(height,width,3).astype(np.float64)/65535
    if role != 'normal':
        if not np.array_equal(rgb[:,:,0],rgb[:,:,1]) or not np.array_equal(rgb[:,:,0],rgb[:,:,2]):
            raise ValueError('Scalar target has unequal RGB channels')
        return rgb[:,:,:1]
    return rgb


def metrics(p,t,role):
    if p.shape != t.shape or not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError('Maps differ in dimensions/channels or contain invalid pixels')
    residual = p-t
    result = {'mae':float(np.abs(residual).mean()), 'mean_error':float(residual.mean()),
              'prediction_range':[float(p.min()),float(p.max())], 'target_range':[float(t.min()),float(t.max())],
              'per_channel_prediction_stddev':p.std(axis=(0,1)).tolist(), 'per_channel_target_stddev':t.std(axis=(0,1)).tolist(),
              'constant_target_mean_baseline_mae':float(np.abs(t-t.mean(axis=(0,1),keepdims=True)).mean())}
    if role != 'normal':
        result['correlation'] = float(np.corrcoef(p.ravel(),t.ravel())[0,1]) if p.std() and t.std() else None
        result['offset_aligned_mae'] = float(np.abs(residual-residual.mean()).mean())
    else:
        pn,tn=p*2-1,t*2-1
        pn=pn/np.maximum(np.linalg.norm(pn,axis=2,keepdims=True),1e-8)
        tn=tn/np.maximum(np.linalg.norm(tn,axis=2,keepdims=True),1e-8)
        result['mean_normal_angle_degrees']=float(np.degrees(np.arccos(np.clip((pn*tn).sum(axis=2),-1,1))).mean())
    result['signed_native_difference_mae']={}
    for offset in [1,4,16,32]:
        errors=[]
        for axis in [0,1]:
            if p.shape[axis] <= offset: continue
            a,b=[slice(None)]*3,[slice(None)]*3
            a[axis],b[axis]=slice(offset,None),slice(None,-offset)
            errors.append(float(np.abs(residual[tuple(a)]-residual[tuple(b)]).mean()))
        if errors: result['signed_native_difference_mae'][str(offset)]=float(np.mean(errors))
    return result


def main():
    path=Path(sys.argv[1]); record=json.loads(path.read_text()); result={'schema':'loom-material-set-metrics-v1','category':record['category'],'maps':{}}
    for role in ['depth','normal','roughness']:
        prediction,target=Path(record[role]),Path(record['original_'+role])
        result['maps'][role]=metrics(read_numeric(prediction,role),read_numeric(target,role),role)
        result['maps'][role]['prediction_sha256']=hashlib.file_digest(prediction.open('rb'),'sha256').hexdigest()
        result['maps'][role]['target_sha256']=hashlib.file_digest(target.open('rb'),'sha256').hexdigest()
    result['note']='Read-only native numeric comparison. These separate metrics do not qualify combined material appearance or generalization.'
    path.with_name(path.stem+'-metrics.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__': main()
