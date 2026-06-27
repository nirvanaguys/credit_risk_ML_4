import joblib
import pandas as pd

model = joblib.load("credit_model.pkl")

data = pd.DataFrame({
    'rev_util':[30],
    'age':[35],
    'late_30_59':[0],
    'debt_ratio':[0.5],
    'monthly_inc':[5000],
    'open_credit':[5],
    'late_90':[0],
    'real_estate':[1],
    'late_60_89':[0],
    'dependents':[1]
})

print("Prediction:", model.predict(data))
print("Probability:", model.predict_proba(data))