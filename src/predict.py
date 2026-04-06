import joblib
import pandas as pd

# Load model
model = joblib.load("models/best_model.pkl")

# Example input (same structure as dataset)
sample_data = pd.DataFrame([{
    'age': 39,
    'workclass': 4,
    'education': 9,
    'marital-status': 2,
    'occupation': 3,
    'relationship': 1,
    'race': 4,
    'sex': 1,
    'capital-gain': 0,
    'capital-loss': 0,
    'hours-per-week': 40,
    'native-country': 38
}])

prediction = model.predict(sample_data)

print("Prediction:", prediction)