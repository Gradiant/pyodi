
# Train Config Generation

::: pyodi.apps.train_config_generation

This script returns a train config file which you can use to train your model. It finds a combination of scales and ratios for creating anchors that can fit your ground truth data.


## Example with COCO dataset

Let's run train-config generation with COCO train data.

```bash
pyodi train-config generation COCO_DATASET_PATH
```

By default images and bounding boxes get resized to 1280 x 720 pixels, this can be modified with `input_size` argument.
