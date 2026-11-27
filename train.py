from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import joblib, mlflow, mlflow.sklearn

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("iris-classification")

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

for C in [0.1, 1.0, 10.0]:
    with mlflow.start_run(run_name=f"logreg_C={C}"):
        params = {"C": C, "max_iter": 1000}
        model = LogisticRegression(**params).fit(X_train, y_train)
        pred = model.predict(X_test)

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy_score(y_test, pred))
        mlflow.log_metric("f1", f1_score(y_test, pred, average="macro"))
        mlflow.sklearn.log_model(
            model, "model", registered_model_name="IrisModel"
        )

joblib.dump(model, "model.joblib")