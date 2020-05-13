"""
# Evaluation App

The [`pyodi evaluation`][pyodi.apps.evaluation.evaluation] app can be used to evaluate
the predictions of an object detection dataset.

Example usage:

``` bash
pyodi evaluation "data/COCO/COCO_val2017.json" "data/COCO/COCO_val2017_predictions.json"
```

This app shows the Average Precision for different IoU values and different areas, the
Average Recall for different IoU values and differents maximum detections, and the
Average F1 Score.

An example of the result of executing this app:
```
 Average Precision                        (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.257
 Average Precision                        (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.439
 Average Precision                        (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = -1.000
 Average Precision                        (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.069
 Average Precision                        (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.277
 Average Precision                        (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.426
 Average Recall                           (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.240
 Average Recall                           (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.354
 Average Recall                           (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.376
 Average Recall                           (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.124
 Average Recall                           (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.416
 Average Recall                           (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.585
 Average F1                               (F1) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.761
```

# API REFERENCE
"""
import json
import re
from typing import Optional

import typer
from loguru import logger

from pyodi.coco.cocoeval import COCOeval
from pyodi.coco.utils import load_coco_ground_truth_from_StringIO

app = typer.Typer()


@logger.catch
@app.command()
def evaluation(
    ground_truth_file: str, predictions_file: str, string_to_match: Optional[str] = None
):
    """
    Evaluate the predictions of a dataset.

    Args:
        ground_truth_file: Path to COCO ground truth file.
        predictions_file: Path to COCO predictions file.
        string_to_match: If not None, only images whose file_name match this parameter will be evaluated.
    """

    coco_ground_truth = load_coco_ground_truth_from_StringIO(open(ground_truth_file))
    coco_predictions = coco_ground_truth.loadRes(json.load(open(predictions_file)))

    coco_eval = COCOeval(coco_ground_truth, coco_predictions)

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
