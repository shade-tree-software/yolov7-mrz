import sys

if len(sys.argv) < 3 or len(sys.argv) > 4:
    print()
    print('Finds objects in an image given a file of weights'
          'representing a trained yolov7 neural network.'
          'Numerical object classes are mapped by default to'
          'the standard list of coco object names, but you can'
          'optionally provide a yaml file with your own list'
          'of object names.')
    print()
    print(f'usage: {sys.argv[0]} <image file> <weights file> [objects file]')
    print()
    exit(0)

from hubconf import custom
from PIL import Image
import pprint
import yaml

DEFAULT_OBJECT_CONFIG_FILE = 'data/coco.yaml'

model = custom(path_or_model=sys.argv[2])

with open(sys.argv[3] if len(sys.argv) == 4 else DEFAULT_OBJECT_CONFIG_FILE, 'r') as file:
    names = yaml.safe_load(file)['names']

img = Image.open(sys.argv[1])
raw_preds = model([img]).xyxy[0]
preds = []
for raw_pred in raw_preds.tolist():
    if raw_pred[4] >= 0.5:
        preds.append({
            'name': names[int(raw_pred[5])],
            'bbox': [round(i) for i in raw_pred[0:4]],
            'conf': raw_pred[4]
        })

print(pprint.pprint(preds))