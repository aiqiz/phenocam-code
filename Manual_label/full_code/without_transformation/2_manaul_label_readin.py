import json
import numpy as np
from PIL import Image, ImageDraw

DIR = "../phenocam02/calibration_reference/"
with open(DIR + 'netcam.phenocam2.20250528_110010.json') as f:
    data = json.load(f)

image_path = data['imagePath']
image = Image.open(DIR + image_path)
w, h = image.size

mask = Image.new('RGB', (w, h), (0, 0, 0))  # black background
draw = ImageDraw.Draw(mask)

label_list = sorted(set(shape['label'] for shape in data['shapes']))
label_color_map = {}

def label_to_color(label):
    np.random.seed(abs(hash(label)) % (2**32))
    return tuple(np.random.randint(50, 256, size=3))  # brighter colors

for label in label_list:
    label_color_map[label] = label_to_color(label)

for shape in data['shapes']:
    label = shape['label']
    points = shape['points']
    polygon = [(x, y) for [x, y] in points]
    color = label_color_map[label]
    draw.polygon(polygon, outline=color, fill=color)

mask.save(DIR + 'reference_colormask.png')
