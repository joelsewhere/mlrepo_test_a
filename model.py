import mlflow

from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

X, y = make_regression(random_state=2026)
model = LinearRegression()

with mlflow.start_run() as run:

    model.fit(X, y)

    mlflow.sklearn.log_model(
        sk_model=model,
        input_example=X,
        registered_model_name='sklearn_regression',
        tags={
            "deployed": "true"
            }
        )