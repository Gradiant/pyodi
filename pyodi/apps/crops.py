import typer

from pyodi.apps.crops_join import crops_join
from pyodi.apps.crops_split import crops_split

crops_app = typer.Typer()
crops_app.command("join")(crops_join)
crops_app.command("split")(crops_split)


if __name__ == "__main__":
    crops_app()
