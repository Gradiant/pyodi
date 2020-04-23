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
    ground_truth_file: str,
    predictions_file: str,
    string_to_match: Optional[str] = None,
):

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

    coco_eval.params.imgIds = filtered_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    metric_names = [
        "Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]",
        "Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]",
        "Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]",
        "Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]",
        "Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]",
        "Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]",
    ]
    eval_results = {}
    for n, metric_name in enumerate(metric_names):
        eval_results[metric_name] = float("{:.3f}".format(coco_eval.stats[n]))

    logger.info(eval_results)


if __name__ == "__main__":
    app()
