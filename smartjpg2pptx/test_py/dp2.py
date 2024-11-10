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
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2  # Adjust the threshold to a more reasonable value
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

# Directory containing images
image_dir = "d:/testjpg"
output_dir = "d:/detected_ppts2"
cropped_dir = "d:/detected_ppts_jpg"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(cropped_dir):
    os.makedirs(cropped_dir)

# Priority order of classes
priority_classes = ["truck", "laptop", "tv", "book"]
class_ids = {class_name: MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes.index(class_name)
             for class_name in priority_classes if class_name in MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes}

# Process each image
for image_name in os.listdir(image_dir):
    if image_name.lower().endswith(".jpg"):
        image_path = os.path.join(image_dir, image_name)
        print(image_path)
        image = cv2.imread(image_path)
        image_area = image.shape[0] * image.shape[1]  # Calculate the area of the image

        outputs = predictor(image)
        instances = outputs["instances"].to("cpu")
        pred_classes = instances.pred_classes.numpy()
        pred_boxes = instances.pred_boxes.tensor.numpy()

        # Find the largest area for priority classes
        max_area = 0
        max_box = None
        for class_name, class_id in class_ids.items():
            mask = pred_classes == class_id
            if any(mask):
                boxes = pred_boxes[mask]
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                largest_index = areas.argmax()
                largest_area = areas[largest_index]
                if largest_area > max_area:
                    max_area = largest_area
                    max_box = boxes[largest_index]

        # Check if the largest area is at least 25% of the image area
        if max_box is not None and max_area >= 0.25 * image_area:
            x1, y1, x2, y2 = map(int, max_box)
            cropped_image = image[y1:y2, x1:x2]
            save_path = os.path.join(cropped_dir, image_name)
            cv2.imwrite(save_path, cropped_image)
        else:
            # Save the original image if no box is large enough
            save_path = os.path.join(cropped_dir, image_name)
            cv2.imwrite(save_path, image)

print("Processing complete.")
