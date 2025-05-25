import argparse
import wandb
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Recoger ID si se pasa desde GitHub Actions
parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, default="manual")
args = parser.parse_args()

wandb.init(project="MLOps-Pycon2023", name=f"iris-run-{args.IdExecution}")

# Dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Modelo
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Evaluación
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Log en WandB
wandb.log({"accuracy": accuracy})
