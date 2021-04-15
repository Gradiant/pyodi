from pyodi.apps.coco.coco_merge import coco_merge
from pyodi.apps.coco.coco_split import property_split, random_split

coco_app = {
    "merge": coco_merge,
    "random-split": random_split,
    "property-split": property_split,
}
