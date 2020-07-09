from typing import Dict, List

from PIL import Image


def get_crops_corners(
    image_pil: Image,
    crop_height: int,
    crop_width: int,
    row_overlap: int = 0,
    col_overlap: int = 0,
) -> List[List[int]]:
    """Divides `image_pil` in crops.

    The crops corners will be generated using the `crop_height`, `crop_width`,
    `row_overlap` and `col_overlap` arguments.

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
    while row_max - row_overlap < height:
        col_min = col_max = 0
        row_max = row_min + crop_height
        while col_max - col_overlap < width:
            col_max = col_min + crop_width
            if row_max > height or col_max > width:
                rmax = min(height, row_max)
                cmax = min(width, col_max)
                crops_corners.append(
                    [cmax - crop_width, rmax - crop_height, cmax, rmax]
                )
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

    right = left + width
    bottom = top + height

    if left > crop_corners[2]:
        return False
    if top > crop_corners[3]:
        return False
    if right < crop_corners[0]:
        return False
    if bottom < crop_corners[1]:
        return False

    return True


def filter_annotation_by_area(
    annotation: Dict, new_annotation: Dict, min_area_threshold: float
) -> bool:
    """Check whether cropped annotation area is smaller than minimum area size.

    Args:
        annotation: Single annotation entry in COCO format.
        new_annotation: Single annotation entry in COCO format.
        min_area_threshold: Minimum area threshold ratio.

    Returns:
        True if annotation area is smaller than the minimum area size.
    """
    area = annotation["area"]
    new_area = new_annotation["area"]
    min_area = area * min_area_threshold

    if new_area > min_area:
        return False

    return True


def get_annotation_in_crop(annotation: Dict, crop_corners: List[int]) -> Dict:
    """Translate annotation coordinates to crop coordinates.

    Args:
        annotation (Dict): Single annotation entry in COCO format.
        crop_corners (List[int]): Generated from `get_crop_corners`.

    Returns:
        Dict: Annotation entry with coordinates translated to crop coordinates.
    """
    left, top, width, height = annotation["bbox"]
    right = left + width
    bottom = top + height

    new_left = max(left - crop_corners[0], 0)
    new_top = max(top - crop_corners[1], 0)
    new_right = min(right - crop_corners[0], crop_corners[2] - crop_corners[0])
    new_bottom = min(bottom - crop_corners[1], crop_corners[3] - crop_corners[1])

    new_width = new_right - new_left
    new_height = new_bottom - new_top

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
