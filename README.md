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
