import os
import argparse
import joblib
import wandb
from sklearn.ensemble import RandomForestClassifier

# Argumentos desde GitHub Actions
parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, default="manual")
args = parser.parse_args()

print(f"IdExecution: {args.IdExecution}")

# Crear carpeta donde guardar el modelo
os.makedirs("model", exist_ok=True)

# Configuración del modelo
model_config = {
    "n_estimators": 100,
    "random_state": 42
}

# Inicializar experimento WandB
with wandb.init(
    project="MLOps-Pycon2023",
    name=f"initialize-rf-{args.IdExecution}",
    job_type="initialize-model",
    config=model_config
) as run:
    config = wandb.config

    # Crear modelo sklearn
    model = RandomForestClassifier(**model_config)
    model_path = "model/random_forest_model.pkl"
    joblib.dump(model, model_path)

    # Subir como artifact a WandB
    artifact = wandb.Artifact(
        name="rf-model",
        type="model",
        description="Random Forest sin entrenar",
        metadata=dict(config)
    )
    artifact.add_file(model_path)
    wandb.save(model_path)
    run.log_artifact(artifact)
