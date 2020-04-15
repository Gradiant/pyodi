import json
from collections import defaultdict

import pandas as pd

from loguru import logger
from pycocotools.coco import COCO


def load_coco_ground_truth_from_StringIO(string_io):
    coco_ground_truth = COCO()
    coco_ground_truth.dataset = json.load(string_io)
    coco_ground_truth.createIndex()
    return coco_ground_truth


def coco_ground_truth_to_dfs(coco_ground_truth, max_images=5000):
    df_images = defaultdict(list)
    categories = {
        x["id"]: x["name"] for x in coco_ground_truth["categories"]
    }
    image_id_to_name = {}
    if len(coco_ground_truth["images"]) > max_images:
        logger.warning(
            f"Number of images {len(coco_ground_truth['images'])} exceeds maximum: {max_images}. "
            "All the exceeding images will be ignored"
        )
    for image in coco_ground_truth["images"][:5000]:
        if n > max_images:
        for k, v in image.items():
            df_images[k].append(v)
        image_id_to_name[image["id"]] = image["file_name"]
    df_images = pd.DataFrame(df_images)

    df_annotations = defaultdict(list)
    for annotation in coco_ground_truth["annotations"][:max_images]:
        df_annotations["file_name"].append(image_id_to_name[annotation["image_id"]])
        df_annotations["category"].append(categories[annotation["category_id"]])
        df_annotations["area"].append(annotation["area"])
        df_annotations["col_centroid"].append(int(annotation["bbox"][0] + (annotation["bbox"][2] // 2)))
        df_annotations["row_centroid"].append(int(annotation["bbox"][1] + (annotation["bbox"][3] // 2)))
        df_annotations["width"].append(int(annotation["bbox"][2]))
        df_annotations["height"].append(int(annotation["bbox"][3]))

    df_annotations = pd.DataFrame(df_annotations)    
    return df_images, df_annotations