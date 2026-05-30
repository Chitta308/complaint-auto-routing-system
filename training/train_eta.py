import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv(r"C:\Users\chitt\complaint-auto-routing-system\data\complaints.csv")

X = np.load("data/complaint_embeddings.npy")

y = df["eta_days"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, pred))

joblib.dump(
    model,
    "models/eta.pkl"
)

print("ETA model saved")