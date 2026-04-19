import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

folder_path = "./cleaned_data"

X = []
y = []

for filename in os.listdir(folder_path):

    if not filename.endswith(".csv"):
        continue

    try:
        df = pd.read_csv(
            f"{folder_path}/{filename}"
        )

        if df["temp"].isna().any():
            continue

        temps = df["temp"].tolist()

        if len(temps) != 12:
            continue
        features = temps[:11]
        target = temps[11]

        X.append(features)
        y.append(target)

    except Exception as e:
        print(filename, e)
X = np.array(X)
y = np.array(y)

print("Liczba obserwacji:", len(X))

# test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

#model regresyjny (SSN)
model = MLPRegressor(
    hidden_layer_sizes=(20,),
    activation='relu',
    max_iter=2000,
    random_state=42
)
model.fit(
    X_train,
    y_train
)
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)


mse_train = mean_squared_error(
    y_train,
    pred_train
)
mse_test = mean_squared_error(
    y_test,
    pred_test
)
r2_train = r2_score(
    y_train,
    pred_train
)
r2_test = r2_score(
    y_test,
    pred_test
)
print("\nWYNIKI:")
print("Train MSE:", mse_train)
print("Test MSE:", mse_test)
print("Train R2:", r2_train)
print("Test R2:", r2_test)


lin_reg = LinearRegression()

lin_reg.fit(X_train, y_train)

pred_lr = lin_reg.predict(X_test)

print("\nLINEAR REGRESSION")
print("MSE:", mean_squared_error(y_test, pred_lr))
print("R2:", r2_score(y_test, pred_lr))

rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

rf.fit(X_train, y_train)

pred_rf = rf.predict(X_test)

print("\nRANDOM FOREST")
print("MSE:", mean_squared_error(y_test, pred_rf))
print("R2:", r2_score(y_test, pred_rf))

svr = SVR(
    kernel='rbf',
    C=1.0,
    epsilon=0.1
)

svr.fit(X_train, y_train)

pred_svr = svr.predict(X_test)

print("\nSVR")
print("MSE:", mean_squared_error(y_test, pred_svr))
print("R2:", r2_score(y_test, pred_svr))
