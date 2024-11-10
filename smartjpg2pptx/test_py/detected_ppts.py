import cv2
import torch
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import os

# Setup logger
setup_logger()

# Configuration
cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1  # 设置阈值
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

# Create a predictor using the config
predictor = DefaultPredictor(cfg)

# Directory containing images
image_dir = "d:/testjpg"
output_dir = "d:/detected_ppts2"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Process each image
for image_name in os.listdir(image_dir):
    if image_name.lower().endswith(".jpg"):
        image_path = os.path.join(image_dir, image_name)
        print(image_path)
        image = cv2.imread(image_path)

        outputs = predictor(image)

        # Visualize the detection results on the image
        v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        result_image = out.get_image()[:, :, ::-1]

        # Save the image with detections
        detected_image_path = os.path.join(output_dir, image_name)
        cv2.imwrite(detected_image_path, result_image)

print("Processing complete.")
