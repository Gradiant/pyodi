import typer

from pyodi.apps.coco_merge import coco_merge
from pyodi.apps.coco_split import property_split, random_split

# TODO Split entrypoint and logic/command modules in packages? (issue #138)

coco_app = typer.Typer()
coco_app.command("merge")(coco_merge)
coco_app.command("random-split")(random_split)
coco_app.command("property-split")(property_split)


if __name__ == "__main__":
    coco_app()
