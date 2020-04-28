import typer

from pyodi.apps.train_config_evaluation import train_config_evaluation
from pyodi.apps.train_config_generation import train_config_generation

train_config_app = typer.Typer()
train_config_app.command("evaluation")(train_config_evaluation)
train_config_app.command("generation")(train_config_generation)


if __name__ == "__main__":
    train_config_app()
