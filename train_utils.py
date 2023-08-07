import wandb
import yaml


# Load YAML config file
def load_config(config_path):
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    return config


# Initialize wandb using the config
def init_wandb_from_config(config):
    run = wandb.init(
        entity=config["entity"],
        project=config["project"],
        name=config["name"],
        config=config["parameters"],
    )
    return run


if __name__ == "__main__":
    print("hello")
