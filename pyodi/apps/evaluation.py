"""# Evaluation App.

The [`pyodi evaluation`][pyodi.apps.evaluation.evaluation] app can be used to evaluate
the predictions of an object detection dataset.

Example usage:

``` bash
pyodi evaluation "data/COCO/COCO_val2017.json" "data/COCO/COCO_val2017_predictions.json"
```

This app shows the Average Precision for different IoU values and different areas, the
Average Recall for different IoU values and differents maximum detections.

An example of the result of executing this app:
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.256
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.438
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.263
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.068
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.278
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.422
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.239
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.353
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.375
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.122
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.416
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.586
```
---

# API REFERENCE
"""  # noqa: E501
import json
import re
from typing import Optional

import typer
from loguru import logger
from pycocotools.cocoeval import COCOeval

from pyodi.core.utils import load_coco_ground_truth_from_StringIO

app = typer.Typer()


@logger.catch
@app.command()
def evaluation(
    ground_truth_file: str, predictions_file: str, string_to_match: Optional[str] = None
) -> None:
    """Evaluate the predictions of a dataset.

    Args:
        ground_truth_file: Path to COCO ground truth file.
        predictions_file: Path to COCO predictions file.
        string_to_match: If not None, only images whose file_name match this parameter
            will be evaluated.

    """
    coco_ground_truth = load_coco_ground_truth_from_StringIO(open(ground_truth_file))  # TODO open objects not closed (it's a command though?)
    coco_predictions = coco_ground_truth.loadRes(json.load(open(predictions_file)))

    coco_eval = COCOeval(coco_ground_truth, coco_predictions, "bbox")

    if string_to_match is not None:
        filtered_ids = [
            k
            for k, v in coco_ground_truth.imgs.items()
            if re.match(string_to_match, v["file_name"])
        ]
        logger.info("Number of filtered_ids: {}".format(len(filtered_ids)))
    else:
        filtered_ids = [k for k in coco_ground_truth.imgs.keys()]

    coco_eval.image_ids = filtered_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == "__main__":
    app()
