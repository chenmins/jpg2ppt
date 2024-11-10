from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.model_zoo import model_zoo
import numpy as np



# 设置配置文件
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

# 创建预测器
predictor = DefaultPredictor(cfg)

# 加载图像
image = predictor.read_image("d:/test1/20241104_105448.jpg", format="BGR")

# 进行推理
outputs = predictor(image)


#
#
#
# # 设置配置文件
# cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 设置置信度阈值
# cfg.MODEL.WEIGHT = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
#
# # 创建预测器
# predictor = DefaultPredictor(cfg)
#
# # 加载图像
# from PIL import Image
# image = Image.open("d:/test1/20241104_105448.jpg")
#
# image = np.array(image)  # 将 PIL.Image 转换为 numpy.ndarray
#
# # 进行推理
# outputs = predictor(image)

# 可视化结果
v = Visualizer(image, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
v.save("output.jpg")  # 保存可视化结果