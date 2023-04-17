from hubconf import custom
from PIL import Image
import sys

model = custom(path_or_model='yolov7-tiny-mrz.pt') 

img = Image.open(sys.argv[1])
img_preds = model([img]).xyxy[0]
best_pred = img_preds[0].tolist()
pred_dict = {
    'xyxy': best_pred[0:4],
    'conf': best_pred[4]
}

print(pred_dict)
