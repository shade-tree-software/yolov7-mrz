from hubconf import custom
from PIL import Image
import pathlib

IMAGE_PATH = 'data/test/images'

model = custom(path_or_model='yolov7-tiny-mrz.pt') 

dir = pathlib.Path(IMAGE_PATH)
images = []
for path in dir.glob('*.[jp][pn]g'):
  images.append(Image.open(str(path)))
df = model(images).pandas().xywh
print(df)