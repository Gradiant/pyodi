import typer

from pyodi.apps.coco_merge import coco_merge

coco_app = typer.Typer()
coco_app.command("merge")(coco_merge)


if __name__ == "__main__":
    coco_app()
