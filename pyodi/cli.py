import typer

from pyodi.apps.evaluation import evaluation
from pyodi.apps.ground_truth import ground_truth
from pyodi.apps.train_config import train_config


app = typer.Typer()

app.command()(evaluation)
app.command()(ground_truth)
app.command()(train_config)


if __name__ == "__main__":
    app()
