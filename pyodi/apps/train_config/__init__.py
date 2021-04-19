from pyodi.apps.train_config.train_config_evaluation import train_config_evaluation
from pyodi.apps.train_config.train_config_generation import train_config_generation

train_config_app = {
    "generation": train_config_generation,
    "evaluation": train_config_evaluation,
}
