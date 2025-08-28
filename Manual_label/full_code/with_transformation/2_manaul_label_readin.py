'''
create using labelme -> urn directly in terminal
save json file & readin to create binary mask
'''

import json
import numpy as np
from PIL import Image, ImageDraw

DIR = "/Volumes/Aiqi_02/phenocams/highpark/"
with open(DIR +'reference_roi.json') as f:
    data = json.load(f)

image_path = data['imagePath']
image = Image.open(DIR + image_path)
w, h = image.size

mask = Image.new('L', (w, h), 0)  # mode 'L' = 8-bit pixels, black and white


for shape in data['shapes']:
    points = shape['points']
    polygon = [(x, y) for [x, y] in points]
    ImageDraw.Draw(mask).polygon(polygon, outline=255, fill=255)

mask.save(DIR + 'reference_binary.png')
