import wandb
import yaml


# Load YAML config file
def load_config(config_path):
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    return config

def train_model(model, train_ds, validation_ds, epochs, callbacks):
    model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=epochs,
        callbacks=callbacks,
    )
    return model

# Initialize wandb using the config
def init_wandb_from_config(config):
    run = wandb.init(
        entity=config["entity"],
        project=config["project"],
        name=config["name"],
        config=config["parameters"],
    )
    return run

def check_artifact_exists(project_name, entity_name, artifact_name):
    # Construct artifact URL.
    artifact_url = f"{entity_name}/{project_name}/{artifact_name}"
    try:
        # Try fetching the artifact.
        artifact = wandb.Api().artifact(artifact_url)
        return True
    except wandb.CommError:
        # Artifact does not exist.
        return False

def get_or_train_model(config_name, project_name, entity_name):
    artifact_name = f"{config_name}:latest"
    
    # Check if artifact exists.
    if check_artifact_exists(project_name, entity_name, artifact_name):
        # Download the artifact.
        artifact = wandb.use_artifact(artifact_name)
        # Load the Keras model from artifact.
        model_path = artifact.get_path("model.keras").download()
        model = load_model(model_path)
    else:
        # Train the model.
        model = train_model()
        
        # Save the model locally.
        model_path = os.path.join(wandb.run.dir, "model.keras")
        model.save(model_path)
        
        # Create and upload artifact to wandb.
        artifact = wandb.Artifact(
            config_name,
            type="model",
            description="trained model",
            metadata=dict(config_name=config_name)
        )
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)
    
    return model


if __name__ == "__main__":
    print("hello")
