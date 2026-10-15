"""Assemble a fixed-scale, jointly rendered three-map review without editing maps."""
import json
from pathlib import Path
import sys
from PIL import Image, ImageDraw, ImageFont

path = Path(sys.argv[1])
record = json.loads(path.read_text())
folder = path.parent / (path.stem + '-renders')
for suffix in ['', '-neutral']:
    canvas = Image.new('RGB', (1536, 1370), (25, 29, 32))
    draw = ImageDraw.Draw(canvas)
    try: font = ImageFont.truetype('/System/Library/Fonts/SFNS.ttf', 20)
    except OSError: font = ImageFont.load_default(size=20)
    for row, light in enumerate(['left', 'right']):
        for column, label in enumerate(['source', 'candidate']):
            x, y = column * 768, row * 685
            title = ('Artist targets' if label == 'source' else 'Trained candidate')
            draw.text((x + 16, y + 12), f"{record['category']} | {title} | {light} light", fill=(240, 240, 240), font=font)
            canvas.paste(Image.open(folder / f'{label}-{light}{suffix}.png'), (x, y + 35))
    canvas.save(folder / ('combined-material' + suffix + '.png'))
