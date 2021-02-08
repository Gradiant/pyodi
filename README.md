<h1 align="center">
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

A simple tool for explore your object detection dataset. The goal of this library is to provide simple and intuitive visualizations from your dataset and automatically find the best parameters for generating a specific grid of anchors that can fit you data characteristics

### Installation
```bash
git clone https://github.com/Gradiant/pyodi.git
cd pyodi
python setup.py install
```

See [Contributing guide](.github/CONTRIBUTING.md) for more info.


### Usage

Pyodi includes different applications that can help you to extract the most from your dataset. A classic flow could follow the following steps:

#### 1. Annotation visualization

With pyodi `paint_annotations` you can easily visualize in a beautiful format your object detection dataset. You can also use this function to visualize model predictions if they are in COCO predictions format.

```bash
pyodi paint-annotations \
  ../tiny_coco/annotations/instances_train2017.json \
  ../tiny_coco/images/train2017 \
  ./painted_ground_truth
```

![COCO image with painted annotations](resources/coco_sample_174482.jpg)

#### 2. Ground truth exploration

It is very recommended to intensively explore your dataset before starting training. The analysis of your images and annotation will allow you to optimize aspects as the optimum image input size for your network or the shape distribution of the bounding boxes. You can use `ground_truth` for this task:

```bash
pyodi ground-truth ../tiny_coco/annotations/instances_train2017.json
```

The output of this command shows three different kinds of plots. The first of them contains information related with the shape of the images present in the dataset. **ADD IMAGE DESCRIPTION WHEN FINAL IMAGE IS SELECTED**
![Image shape distribution](resources/gt_img_shapes.png)
We can also observe bounding box distribution, with the possibility of enabling filters by class or sets of classes. **This dataset shows a clear tendency to  rectangular bounding boxes with larger width than height and where most of them embrace areas below the 20% of the total image.**
![Bbox distribution](resources/gt_bb_shapes.png)

Finally, we can also check where the centers of bounding boxes are most commonly found with respect to the image, which can help us distinguish ROIs in input images. In this case we observe that the objects usually appear in the upper half of the images.
![Bbox center distribution](resources/gt_bb_centers.png)

#### 3. Train config generation

The design of anchors is critical for the performance of one-stage detectors. Usually, published models such [RetinaNet](https://arxiv.org/abs/1708.02002) include default anchors which has been designed to work with general object detection purpose as COCO dataset. Sometimes you work with data which contains only a few different classes that share similar properties as the bounding box shape, this would be the case for a drone detection dataset such [Drone vs Bird](https://wosdetc2020.wordpress.com/). You can exploit this by designing anchors that specially fit the distribution of your data, optimizing the probability of matching ground truth bounding boxes with generated anchors, which can result in an increase in the performance of your model. At the same time, you can reduce the number of anchors you use to boost inference and training time.

With pyodi `train-config generation` you can automatically find a set of anchors that fit your data distribution. We can adjust the number of scales and ratios we want and the starting anchor base size for each pyramid level. The input size determines the model input size and automatically adjusts images and annotations shapes to it.

```bash
pyodi train-config generation \
  ../tiny_coco/annotations/instances_train2017.json \
  --input-size 1280 70 \
  --n-ratios 3 \
  --n-scales 3 \
  --output ./resources
```

Result of this command shows two different plots. The left plot log scale vs log ratio of bounding boxes and proposed anchors and the right one compares scaled widths and heights. It is easy to observe how proposed anchors follow original data distribution.

![Anchor clustering plot](resources/clusters.png)

Proposed anchors are also attached in a Json file that follows [mmdetection anchors](https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/anchor/anchor_generator.py#L10) format:
```python
anchor_config=dict(
    type='AnchorGenerator',
    scales=[0.14, 0.42, 0.91],
    ratios=[0.03, 0.10, 0.22],
    strides=[4, 8, 16, 32, 64],
    base_sizes=[32, 64, 128, 256, 512],
)
```
