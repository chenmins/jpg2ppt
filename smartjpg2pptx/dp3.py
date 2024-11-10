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
import shutil

# Setup logger
setup_logger()

# Configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1  # Adjust the threshold here
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create a predictor using the config
predictor = DefaultPredictor(cfg)

# Directory containing images
image_dir = "d:/jpg"
output_dir = "d:/detected_ppts"
crop_dir = "d:/detected_ppts_jpg"

# Clear existing files
for directory in [output_dir, crop_dir]:
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

# Target classes
target_classes = ["truck", "laptop", "tv", "book"]

# Process each image
for image_name in os.listdir(image_dir):
    if image_name.lower().endswith(".jpg"):
        image_path = os.path.join(image_dir, image_name)
        print(f"Processing {image_path}")
        image = cv2.imread(image_path)

        outputs = predictor(image)
        instances = outputs["instances"].to("cpu")

        # Filter instances for target classes and find the largest
        pred_classes = instances.pred_classes
        pred_boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores
        class_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
        areas = [(x[2] - x[0]) * (y[3] - y[1]) for x, y in zip(pred_boxes, pred_boxes)]

        largest_area = 0
        largest_index = -1

        for i, (cls, area) in enumerate(zip(pred_classes, areas)):
            class_name = class_names[cls]
            if class_name in target_classes and area > largest_area:
                largest_area = area
                largest_index = i

        if largest_index >= 0:
            # Visualize and save the image with detections
            v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            out = v.draw_instance_predictions(instances[largest_index:largest_index+1])
            result_image = out.get_image()[:, :, ::-1]
            detected_image_path = os.path.join(output_dir, image_name)
            cv2.imwrite(detected_image_path, result_image)

            # Crop the largest detected object
            box = pred_boxes[largest_index]
            x1, y1, x2, y2 = map(int, box)
            cropped_image = image[y1:y2, x1:x2]
            if (x2 - x1) * (y2 - y1) < 0.25 * image.shape[0] * image.shape[1]:
                print(f"Warning: Small object in {image_name}, saving original image.")
                cv2.imwrite(os.path.join(crop_dir, image_name), image)
            else:
                cv2.imwrite(os.path.join(crop_dir, image_name), cropped_image)
        else:
            print(f"No target objects found in {image_name}, saving original image.")
            cv2.imwrite(os.path.join(crop_dir, image_name), image)

print("Processing complete.")
