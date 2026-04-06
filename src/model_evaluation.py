from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def evaluate_models(models, X_test, y_test):

    results = {}
    best_model = None
    best_score = -float("inf")
    best_name = ""

    # Detect problem type
    if y_test.nunique() < 20:
        problem_type = "classification"
    else:
        problem_type = "regression"

    for name, model in models.items():

        preds = model.predict(X_test)

        if problem_type == "classification":
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average='weighted')

            results[name] = {
                "accuracy": acc,
                "f1": f1
            }

            score = f1  # choose best based on F1

        else:
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)

            results[name] = {
                "rmse": rmse,
                "r2": r2
            }

            score = r2  # choose best based on R2

        if score > best_score:
            best_score = score
            best_model = model
            best_name = name

    return results, best_model, best_name, best_score, problem_type