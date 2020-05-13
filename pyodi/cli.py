import typer

from pyodi.apps.crops import crops_app
from pyodi.apps.evaluation import evaluation
from pyodi.apps.ground_truth import ground_truth
from pyodi.apps.train_config import train_config_app

app = typer.Typer()

app.command()(evaluation)
app.command()(ground_truth)
app.add_typer(train_config_app, name="train-config")
app.add_typer(crops_app, name="crops")


if __name__ == "__main__":
    app()
