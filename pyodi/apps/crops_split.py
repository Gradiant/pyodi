import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import typer
from loguru import logger
from PIL import Image

app = typer.Typer()


def get_crops_corners(
    image_pil: Image,
    crop_height: int,
    crop_width: int,
    row_overlap: int = 0,
    col_overlap: int = 0,
) -> List[List[int]]:
    """Divide `image_pil` in crops and return a list of corner coordinates for each crop.

    The crops corners will be generated using the `crop_height`, `crop_width`, `row_overlap` and `col_overlap` arguments.

    Args:
        image_pil (PIL.Image): Instance of PIL.Image
        crop_height (int)
        crop_width (int)
        row_overlap (int, optional): Default 0.
        col_overlap (int, optional): Default 0.

    Returns:
        List[List[int]]: List of 4 corner coordinates for each crop of the N crops.
            [
                [crop_0_left, crop_0_top, crop_0_right, crop_0_bottom],
                ...
                [crop_N_left, crop_N_top, crop_N_right, crop_N_bottom]
            ]
    """
    crops_corners = []
    row_max = row_min = 0
    width, height = image_pil.size
    while row_max < height:
        col_min = col_max = 0
        row_max = row_min + crop_height
        while col_max < width:
            col_max = col_min + crop_width
            if row_max > crop_height or col_max > crop_width:
                rmax = min(crop_height, row_max)
                cmax = min(crop_width, col_max)
                crops_corners.append([cmax - width, rmax - height, cmax, rmax])
            else:
                crops_corners.append([col_min, row_min, col_max, row_max])
            col_min = col_max - col_overlap
        row_min = row_max - row_overlap
    return crops_corners


def annotation_inside_crop(annotation: Dict, crop_corners: List[int]) -> bool:
    """Check whether annotation coordinates lie inside crop coordinates.

    Args:
        annotation (Dict): Single annotation entry in COCO format.
        crop_corners (List[int]): Generated from `get_crop_corners`.

    Returns:
        bool: True if any annotation coordinate lies inside crop.
    """
    left, top, width, height = annotation["bbox"]

    if (
        left < crop_corners[2]
        or top < crop_corners[3]
        or left + width > crop_corners[0]
        or top + height > crop_corners[1]
    ):
        return True

    return False


def get_annotation_in_crop(annotation: Dict, crop_corners: List[int]) -> Dict:
    """Translate annotation coordinates to crop coordinates.

    Args:
        annotation (Dict): Single annotation entry in COCO format.
        crop_corners (List[int]): Generated from `get_crop_corners`.

    Returns:
        Dict: Annotation entry with coordinates translated to crop coordinates.
    """
    left, top, width, height = annotation["bbox"]

    new_left = min(left - crop_corners[0], 0)
    new_top = min(top - crop_corners[1], 0)
    if new_left + width > crop_corners[2]:
        new_width = crop_corners[2] - new_left
    else:
        new_width = width

    if new_top + height > crop_corners[3]:
        new_height = crop_corners[3] - new_top
    else:
        new_height = height

    new_bbox = [new_left, new_top, new_width, new_height]
    new_area = new_width * new_height
    new_segmentation = [
        new_left,
        new_top,
        new_left,
        new_top + new_height,
        new_left + new_width,
        new_top + new_height,
        new_left + new_width,
        new_top,
    ]
    return {
        "bbox": new_bbox,
        "area": new_area,
        "segmentation": new_segmentation,
        "iscrowd": annotation["iscrowd"],
        "score": annotation["score"],
        "category_id": annotation["category_id"],
    }


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
        image_pil = Image.open(Path(image_folder) / file_name)

        crops_corners = get_crops_corners(
            image_pil, crop_height, crop_width, row_overlap, col_overlap
        )

        for crop_corners in crops_corners:
            crop = image_pil.crop(crop_corners)

            crop_file_name = (
                output_image_folder_path
                / f"{file_name}_{crop_corners[0]}_{crop_corners[1]}{file_name.suffix}"
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
