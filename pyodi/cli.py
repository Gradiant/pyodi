import fire

from pyodi.apps.coco import coco_app
from pyodi.apps.crops import crops_app
from pyodi.apps.evaluation import evaluation
from pyodi.apps.ground_truth import ground_truth
from pyodi.apps.paint_annotations import paint_annotations
from pyodi.apps.train_config import train_config_app


def app() -> None:
    """Cli app."""
    fire.Fire(
        {
            "evaluation": evaluation,
            "ground_truth": ground_truth,
            "paint_annotations": paint_annotations,
            "train-config": train_config_app,
            "crops": crops_app,
            "coco": coco_app,
        }
    )


if __name__ == "__main__":
    app()
