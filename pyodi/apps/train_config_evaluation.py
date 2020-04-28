import os.path as osp
import sys

from importlib import import_module
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from typing import Optional

import typer

from loguru import logger

from pyodi.coco.utils import (
    coco_ground_truth_to_dfs,
    load_ground_truth_file
)
from pyodi.core.anchor_generator import AnchorGenerator


app = typer.Typer()


def load_train_config_file(train_config_file: str) -> dict:
    logger.info("Loading Train Config File")
    with TemporaryDirectory() as temp_config_dir:
        copyfile(train_config_file,
                        osp.join(temp_config_dir, '_tempconfig.py'))
        sys.path.insert(0, temp_config_dir)
        mod = import_module('_tempconfig')
        sys.path.pop(0)
        train_config = {
            name: value
            for name, value in mod.__dict__.items()
            if not name.startswith('__')
        }
        # delete imported module
        del sys.modules['_tempconfig']
    return train_config


@logger.catch
@app.command()
def train_config_evaluation(
    ground_truth_file: str,
    train_config_file: str,
    show: bool = True,
    output: Optional[str] = None,
):
    """Evaluates the fitness between `ground_truth_file` and `train_config_file`.

    Parameters
    ----------
    ground_truth_file : str
        Path to COCO ground truth file
    train_config_file: str
        Path to MMDetection-like configuration file.
        Must contain `train_pipeline` and `anchor_generator` sections.

        Example content:

            ```python
            # faster_rcnn_r50_fpn_gn_ws_config.py
            train_pipeline = [
                dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(type='Pad', size_divisor=32)
            ]
            anchor_generator=dict(
                type='AnchorGenerator',
                scales=[8],
                ratios=[0.5, 1.0, 2.0],
                strides=[4, 8, 16, 32, 64]
            )
            ```
    show : bool, optional
        Show results or not, by default True
    output : str, optional
        Output file where results are saved, by default None
    """

    if output is not None:
        output = str(Path(output) / Path(ground_truth_file).name)

    coco_ground_truth = load_ground_truth_file(ground_truth_file)

    df_images, df_annotations = coco_ground_truth_to_dfs(coco_ground_truth)

    train_config = load_train_config_file(train_config_file)
    
    del train_config["anchor_generator"]["type"]
    anchor_generator = AnchorGenerator(**train_config["anchor_generator"])
    
    logger.info(anchor_generator)



if __name__ == "__main__":
    app()
