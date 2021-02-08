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

#### 1. Ground truth visualization
When dealing with a computer vision dataset, one of the first things one must do is to have a look at how data looks and pyodi `paint_annotations` is perfect for this.
```bash
pyodi paint-annotations \
  ../tiny_coco/annotations/instances_train2017.json \
  ../tiny_coco/images/train2017 \
  ./painted_ground_truth
```

[ADD_IMAGE]

