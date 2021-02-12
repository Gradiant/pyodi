# Welcome to Python Object Detection Insights

A library for exploring your object detection dataset. The goal of this library is to provide simple and intuitive visualizations from your dataset and automatically find the best parameters for generating a specific grid of anchors that can fit you data characteristics
## Commands

* [`pyodi paint-annotations`](reference/apps/paint-annotations.md) - Paint COCO format annotations and predictions
* [`pyodi ground-truth`](reference/apps/ground-truth.md) - Explore your dataset ground truth characteristics.
* [`pyodi evaluation`](reference/apps/evaluation.md) - Evaluate the predictions of your model against your ground truth.
* [`pyodi train-config generation`](reference/apps/train-config-generation.md) - Automatically generate a `train_config_file` using `ground_truth_file`.
* [`pyodi train-config evaluation`](reference/apps/train-config-evaluation.md) - Evaluate the fitness between `ground_truth_file` and `train_config_file`..
* [`pyodi coco merge`](reference/apps/coco-merge.md) - Automatically merge COCO annotation files.
* [`pyodi coco split`](reference/apps/coco-split.md) - Creates a new dataset by splitting images into crops and adapting the annotations file
* [`pyodi crops split`](reference/apps/crops-split.md) - Creates a new dataset by splitting images into crops and adapting the annotations file
* [`pyodi crops merge`](reference/apps/crops-merge.md) - Translate COCO ground truth or COCO predictions crops split into original image coordinates
