import os
import cv2
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

# 设置路径
input_dir = r'd:\test1'
output_dir = r'd:\newjpg'

# 创建输出文件夹（如果不存在）
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 配置Detectron2模型
cfg = get_cfg()

# 从Detectron2 Model Zoo获取配置文件和权重
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

# 设置权重
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl")
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # 设置阈值
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

# 设置阈值和设备（GPU 或 CPU）
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 设置阈值
cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 创建预测器
predictor = DefaultPredictor(cfg)

# 处理每一张图片
for filename in os.listdir(input_dir):
    if filename.endswith('.jpg'):
        img_path = os.path.join(input_dir, filename)
        image = cv2.imread(img_path)

        # 进行推理
        outputs = predictor(image)
        instances = outputs['instances']

        # 获取检测结果
        pred_classes = instances.pred_classes  # 类别标签
        pred_boxes = instances.pred_boxes.tensor.cpu().numpy()  # 边界框

        # 假设幻灯片属于某个特定类别（你可以根据实际情况调整）
        # 这里假设类别 '33' 代表幻灯片（你可以根据实际类别调整）
        for i in range(len(pred_classes)):
            if pred_classes[i] == 73:  # 假设幻灯片的类别ID是33
                box = pred_boxes[i]
                x1, y1, x2, y2 = box
                cropped_image = image[int(y1):int(y2), int(x1):int(x2)]

                # 保存裁剪的图像
                output_filename = f"{os.path.splitext(filename)[0]}_slide_{i}.jpg"
                output_path = os.path.join(output_dir, output_filename)
                cv2.imwrite(output_path, cropped_image)

        print(f"Processed {filename}")
