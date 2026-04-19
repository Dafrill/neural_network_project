n_estimators_list = [10, 50, 100, 200]

for n in n_estimators_list:

    rf = RandomForestRegressor(
        n_estimators=n,
        max_depth=10,
        random_state=42
    )

    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)

    print("n_estimators:", n)
    print("MSE:", mean_squared_error(y_test, pred))
    print("R2:", r2_score(y_test, pred))

    C_values = [0.1, 1, 10, 100]
kernels = ["linear", "rbf", "poly", "sigmoid"]

for c in C_values:

    for k in kernels:

        svr = SVR(C=c, kernel=k)

        svr.fit(X_train, y_train)
        pred = svr.predict(X_test)

        print("C:", c, "kernel:", k)
        print("MSE:", mean_squared_error(y_test, pred))
        print("R2:", r2_score(y_test, pred))
        neurons = [5, 10, 20, 50]

for n in neurons:
    mlp = MLPRegressor(
        hidden_layer_sizes=(n,),
        max_iter=2000,
        random_state=42
    )
    mlp.fit(X_train, y_train)
    pred = mlp.predict(X_test)

    print("neurons:", n)
    print("MSE:", mean_squared_error(y_test, pred))
    print("R2:", r2_score(y_test, pred))
