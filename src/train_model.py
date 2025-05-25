import wandb
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

wandb.init(project="mlops-pycon2023", name="iris-classification")

# Cargar datos
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)

# Entrenar modelo
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Evaluar
acc = accuracy_score(y_test, y_pred)
wandb.log({"accuracy": acc})
