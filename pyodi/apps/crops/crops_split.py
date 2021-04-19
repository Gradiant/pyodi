"""# Crops Split App.

---

# API REFERENCE
"""
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from loguru import logger
from PIL import Image

from pyodi.core.crops import (
    annotation_inside_crop,
    filter_annotation_by_area,
    get_annotation_in_crop,
    get_crops_corners,
)


@logger.catch
def crops_split(
    ground_truth_file: str,
    image_folder: str,
    output_file: str,
    output_image_folder: str,
    crop_height: int,
    crop_width: int,
    row_overlap: int = 0,
    col_overlap: int = 0,
    min_area_threshold: float = 0.0,
) -> None:
    """Creates new dataset by splitting images into crops and adapting the annotations.

    Args:
        ground_truth_file: Path to a COCO ground_truth_file.
        image_folder: Path where the images of the ground_truth_file are stored.
        output_file: Path where the `new_ground_truth_file` will be saved.
        output_image_folder: Path where the crops will be saved.
        crop_height: Crop height.
        crop_width: Crop width
        row_overlap: Row overlap. Defaults to 0.
        col_overlap: Column overlap. Defaults to 0.
        min_area_threshold: Minimum area threshold ratio. If the cropped annotation area
            is smaller than the threshold, the annotation is filtered out. Defaults to 0.

    """
    ground_truth = json.load(open(ground_truth_file))

    image_id_to_annotations: Dict = defaultdict(list)
    for annotation in ground_truth["annotations"]:
        image_id_to_annotations[annotation["image_id"]].append(annotation)

    output_image_folder_path = Path(output_image_folder)
    output_image_folder_path.mkdir(exist_ok=True, parents=True)

    new_images: List = []
    new_annotations: List = []

    for image in ground_truth["images"]:

        file_name = Path(image["file_name"])
        logger.info(file_name)
        image_pil = Image.open(Path(image_folder) / file_name.name)

        crops_corners = get_crops_corners(
            image_pil, crop_height, crop_width, row_overlap, col_overlap
        )

        for crop_corners in crops_corners:
            logger.info(crop_corners)
            crop = image_pil.crop(crop_corners)

            crop_suffixes = "_".join(map(str, crop_corners))
            crop_file_name = f"{file_name.stem}_{crop_suffixes}{file_name.suffix}"

            crop.save(output_image_folder_path / crop_file_name)

            crop_id = len(new_images)
            new_images.append(
                {
                    "file_name": Path(crop_file_name).name,
                    "height": int(crop_height),
                    "width": int(crop_width),
                    "id": int(crop_id),
                }
            )
            for annotation in image_id_to_annotations[image["id"]]:
                if not annotation_inside_crop(annotation, crop_corners):
                    continue
                new_annotation = get_annotation_in_crop(annotation, crop_corners)
                if filter_annotation_by_area(
                    annotation, new_annotation, min_area_threshold
                ):
                    continue
                new_annotation["id"] = len(new_annotations)
                new_annotation["image_id"] = crop_id
                new_annotations.append(new_annotation)

    new_ground_truth = {
        "images": new_images,
        "old_images": ground_truth["images"],
        "annotations": new_annotations,
        "categories": ground_truth["categories"],
        "licenses": ground_truth.get("licenses", []),
        "info": ground_truth.get("info"),
    }

    with open(output_file, "w") as f:
        json.dump(new_ground_truth, f, indent=2)
