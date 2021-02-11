## Installation

### Requirements

- Python 3.7+


### Install pyodi

a. Create a conda virtual environment and activate it.

```bash
conda create -n pyodi python=3.7 -y
conda activate pyodi
```

b. Clone the pyodi repository.

```bash
git clone https://github.com/pyodi/pyodi.git
cd pyodi
```

c. Install build requirements and then install pyodi.
(We install a forked version of pycocotools via the github repo instead of pypi
for better compatibility with our repo.)

```bash
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
```

Note: Some dependencies are only necessary for development. Simply running `pip install -v -e .` will only install the minimum runtime requirements. If you
want to install the other dependencies, either install them manually with `pip install -r requirements/dev.txt` or specify it in the desired extras when
calling `pip`, that is `pip install -v -e .[all]`.
