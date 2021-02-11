
<h1 align="center">
  <div>
    <img style="max-width: 65px" src="docs/images/logo.svg" >
  </div>
  <b>Pyodi</b><br>
</h1>


<h3 align="center">
  <b>Python Object Detection Insights</b><br>
</h3>

<p align="center">
    <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/Python-3.7-ff69b4.svg" />
    </a>
    <a href="https://github.com/Gradiant/pyodi/actions?query=workflow%3A%22Continuous+Integration%22">
        <img src="https://github.com/pyodi/pyodi/workflows/Continuous%20Integration/badge.svg?branch=master" />
    </a>
    <a href= "https://github.com/AntixK/PyTorch-VAE/blob/master/LICENSE.md">
        <img src="https://img.shields.io/badge/license-Apache2.0-blue.svg" />
    </a>
</p>

```bash
pip install pyodi
```

## Documentation: https://Gradiant.github.io/pyodi

## Quickstart

`pyodi` includes different applications that can help you to extract the most from your dataset. You can download our `TINY_COCO_ANIMAL` dataset from the [releases page](https://github.com/Gradiant/pyodi/releases/tag/v0.1.0) in order to test the example commands. A classic flow could follow the following steps:


```bash
pyodi paint-annotations \
$TINY_COCO_ANIMAL/annotations/train.json \
$TINY_COCO_ANIMAL/sample_images/ \
$TINY_COCO_ANIMAL/painted_images/
```

![COCO image with painted annotations](docs/images/coco_sample_82680.jpg)

```bash
pyodi ground-truth \
$TINY_COCO_ANIMAL/annotations/train.json
```

![Image shape distribution](docs/images/ground_truth/gt_img_shapes.png)

![Bbox distribution](docs/images/ground_truth/gt_bb_shapes.png)

![Bbox center distribution](docs/images/ground_truth/gt_bb_centers.png)

```bash
pyodi train-config generation \
$TINY_COCO_ANIMAL/annotations/train.json \
--input-size 1280 720 \
--n-ratios 3 --n-scales 3
```

![Anchor clustering plot](docs/images/train-config-generation/clusters.png)


```bash
pyodi train-config evaluation \
$TINY_COCO_ANIMAL/annotations/train.json \
$TINY_COCO_ANIMAL/resources/anchor_config.py \
--input-size 1280 720
```

![Anchor overlap plot](docs/images/train-config-evaluation/overlap.png)

## Contributing

We appreciate all contributions to improve Pyodi. Please refer to [Contributing guide](.github/CONTRIBUTING.md) for more info.
