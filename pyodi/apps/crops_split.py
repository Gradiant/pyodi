import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import typer
from loguru import logger

app = typer.Typer()


@logger.catch
@app.command()
def crops_split(ground_truth, predictions, output):
    pass


if __name__ == "__main__":
    app()
