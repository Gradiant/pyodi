"""# Evaluation App.

The [`pyodi evaluation`][pyodi.apps.evaluation.evaluation] app can be used to evaluate
the predictions of an object detection dataset.

Example usage:

``` bash
pyodi evaluation "data/COCO/COCO_val2017.json" "data/COCO/COCO_val2017_predictions.json"
```

This app shows the Average Precision for different IoU values and different areas, the
Average Recall for different IoU values and differents maximum detections, and the
Optimal LRP values for IoU 0.5.

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
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.784
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.206
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.363
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.579
# Class-specific LRP-Optimal Thresholds # 
 [0.256 0.215 0.278 0.246 0.434 0.378 0.441 0.218 0.228 0.181 0.328 0.434
 0.28  0.315 0.181 0.277 0.405 0.288 0.304 0.324 0.342 0.48  0.325 0.355
 0.14  0.246 0.132 0.18  0.246 0.176 0.161 0.16  0.197 0.194 0.226 0.172
 0.244 0.205 0.269 0.177 0.146 0.242 0.148 0.13  0.136 0.208 0.23  0.197
 0.194 0.255 0.221 0.18  0.179 0.316 0.196 0.187 0.206 0.344 0.206 0.262
 0.357 0.473 0.337 0.432 0.286 0.192 0.346 0.137 0.272 0.305 0.038 0.243
 0.295 0.162 0.329 0.199 0.237 0.366   nan 0.139]
```

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
    coco_ground_truth = load_coco_ground_truth_from_StringIO(open(ground_truth_file))
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
