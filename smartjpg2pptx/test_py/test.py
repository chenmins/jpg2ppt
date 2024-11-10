from detectron2.data import MetadataCatalog

# 获取COCO的类别信息
metadata = MetadataCatalog.get("coco_2017_val")
categories = metadata.thing_classes

for idx, category in enumerate(categories):
    print(f"ID: {idx}, Category: {category}")
