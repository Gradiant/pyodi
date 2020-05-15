import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import typer
from loguru import logger
from PIL import Image

from pyodi.core.crops import (
    annotation_inside_crop,
    get_annotation_in_crop,
    get_crops_corners,
)

app = typer.Typer()


@logger.catch
@app.command()
def crops_split(
    ground_truth_file: str,
    image_folder: str,
    output_file: str,
    output_image_folder: str,
    crop_height: int,
    crop_width: int,
    row_overlap: int = 0,
    col_overlap: int = 0,
) -> None:
    """Generate a new dataset by splitting the images into crops and adapting the annotations.

    Args:
        ground_truth_file (str): Path to a COCO ground_truth_file.
        image_folder (str): Path where the images of the ground_truth_file are stored.
        output_file (str): Path where the `new_ground_truth_file` will be saved.
        output_image_folder (str): Path where the crops will be saved.
        crop_height (int)
        crop_width (int)
        row_overlap (int, optional): Default 0.
        col_overlap (int, optional): Default 0.

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

        file_name = image["file_name"]
        logger.info(file_name)
        image_pil = Image.open(Path(image_folder) / file_name)

        crops_corners = get_crops_corners(
            image_pil, crop_height, crop_width, row_overlap, col_overlap
        )

        for crop_corners in crops_corners:
            logger.info(crop_corners)
            crop = image_pil.crop(crop_corners)

            crop_file_name = (
                output_image_folder_path
                / f"{Path(file_name).stem}_{crop_corners[0]}_{crop_corners[1]}{Path(file_name).suffix}"
            )
            crop.save(crop_file_name)

            crop_id = len(new_images)
            new_images.append(
                {
                    "file_name": crop_file_name,
                    "height": crop_height,
                    "width": crop_width,
                    "id": crop_id,
                }
            )
            for annotation in image_id_to_annotations[image["id"]]:
                if not annotation_inside_crop(annotation, crop_corners):
                    continue
                new_annotation = get_annotation_in_crop(annotation, crop_corners)
                new_annotation["id"] = len(new_annotations)
                new_annotation["image_id"] = crop_id
                new_annotations.append(new_annotation)

    new_ground_truth = {
        "images": new_images,
        "old_images": ground_truth["images"],
        "annotations": new_annotations,
        "categories": ground_truth["categories"],
        "licenses": ground_truth["licenses"],
        "info": ground_truth["info"],
    }

    with open(output_file, "w") as f:
        json.dump(new_ground_truth, f, indent=2)


if __name__ == "__main__":
    app()
