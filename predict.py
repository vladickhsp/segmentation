import os
import torch
import cv2
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
import matplotlib.pyplot as plt
from datetime import datetime

# Підготовка до завантаження наборів даних
DATASET_PATH = "/Users/vladislavtruhanovskiy/Desktop/log/dataset"

# TRAIN SET
TRAIN_DATA_SET_NAME = "tree-train"
TRAIN_DATA_SET_IMAGES_DIR_PATH = os.path.join(DATASET_PATH, "train")
TRAIN_DATA_SET_ANN_FILE_PATH = os.path.join(DATASET_PATH, "train", "_annotations.coco.json")

register_coco_instances(
    name=TRAIN_DATA_SET_NAME,
    metadata={},
    json_file=TRAIN_DATA_SET_ANN_FILE_PATH,
    image_root=TRAIN_DATA_SET_IMAGES_DIR_PATH
)

# VALID SET
VALID_DATA_SET_NAME = "tree-valid"
VALID_DATA_SET_IMAGES_DIR_PATH = os.path.join(DATASET_PATH, "valid")
VALID_DATA_SET_ANN_FILE_PATH = os.path.join(DATASET_PATH, "valid", "_annotations.coco.json")

register_coco_instances(
    name=VALID_DATA_SET_NAME,
    metadata={},
    json_file=VALID_DATA_SET_ANN_FILE_PATH,
    image_root=VALID_DATA_SET_IMAGES_DIR_PATH
)

# TEST SET
TEST_DATA_SET_NAME = "tree-test"
TEST_DATA_SET_IMAGES_DIR_PATH = os.path.join(DATASET_PATH, "test")
TEST_DATA_SET_ANN_FILE_PATH = os.path.join(DATASET_PATH, "test", "_annotations.coco.json")

register_coco_instances(
    name=TEST_DATA_SET_NAME,
    metadata={},
    json_file=TEST_DATA_SET_ANN_FILE_PATH,
    image_root=TEST_DATA_SET_IMAGES_DIR_PATH
)

# Підготовка конфігурації моделі
cfg = get_cfg()
cfg.merge_from_file("/Users/vladislavtruhanovskiy/Desktop/log/mask_rcnn_R_101_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = "/Users/vladislavtruhanovskiy/Desktop/log/model_final.pth"  # Шлях до збережених ваг
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
cfg.MODEL.DEVICE = "cpu"  # Використовуємо CPU

# Створення предиктора
predictor = DefaultPredictor(cfg)

# Виконання передбачення
image_path = "/Users/vladislavtruhanovskiy/Desktop/log/result_dyn_646004_l.jpg"
im = cv2.imread(image_path)
outputs = predictor(im)

# Створення маски
masks = outputs["instances"].pred_masks.to("cpu").numpy()
binary_mask = (masks.sum(axis=0) > 0).astype('uint8') * 255
mask_output_path = '/Users/vladislavtruhanovskiy/Desktop/log/predict/binary_mask.jpg'
cv2.imwrite(mask_output_path, binary_mask)

# Візуалізація за допомогою Matplotlib
v = Visualizer(im[:, :, ::-1], scale=0.8, instance_mode=ColorMode.IMAGE_BW)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

plt.imshow(out.get_image()[:, :, ::-1])
plt.title("Predictions")
plt.show()

plt.imshow(binary_mask, cmap="gray")
plt.title("Binary Mask")
plt.show()
