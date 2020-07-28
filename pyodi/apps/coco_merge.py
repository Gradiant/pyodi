import json
from pathlib import Path

import typer
from loguru import logger

app = typer.Typer()


@logger.catch
@app.command()
def coco_merge(output_file, annotation_files=[], base_imgs_folders=[]):
    return