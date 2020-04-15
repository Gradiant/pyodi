import json
import re

import streamlit as st

from coco.cocoeval import COCOeval
from coco.utils import load_coco_ground_truth_from_StringIO


st.title("Object Detection Insights")

st.header("Evaluation")

ground_truth_file = st.sidebar.file_uploader("COCO Ground Truth File", type="json")
predictions_file = st.sidebar.file_uploader("COCO Predictions File", type="json")

string_to_match = st.sidebar.text_input("String to match", value="drone_racing")

if ground_truth_file is not None and predictions_file is not None:
    coco_ground_truth = load_coco_ground_truth_from_StringIO(ground_truth_file)
    coco_predictions = coco_ground_truth.loadRes(json.load(predictions_file)["annotations"])

    coco_eval = COCOeval(coco_ground_truth, coco_predictions, "bbox")

    filtered_ids = [
        k for k, v in coco_ground_truth.imgs.items() 
        if re.match(string_to_match, v["file_name"])
    ]
    st.write("Number of filtered_ids: {}".format(len(filtered_ids)))

    coco_eval.params.imgIds = filtered_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    metric_names = [
        'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]', 
        'Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]', 
        'Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]', 
        'Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]', 
        'Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
        'Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]'
    ]
    eval_results = {}
    for n, metric_name in enumerate(metric_names):
        eval_results[metric_name] =  float('{:.3f}'.format(coco_eval.stats[n]))

    st.write(eval_results)
