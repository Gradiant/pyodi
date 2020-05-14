import typer

from pyodi.apps.crops_merge import crops_merge
from pyodi.apps.crops_split import crops_split

crops_app = typer.Typer()
crops_app.command("merge")(crops_merge)
crops_app.command("split")(crops_split)


if __name__ == "__main__":
    crops_app()
