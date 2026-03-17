import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import RepeatedKFold
from keras.models import Sequential
from keras.layers import Dense, Input
from keras import optimizers
from matplotlib import pyplot as plt


def set_seed(seed=22527):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_dataset(file_path="butan(gas)_db.csv"):
    data = pd.read_csv(file_path, sep=";", decimal=",")
    data = data[["Temperature", "Pressure", "Density"]].dropna().reset_index(drop=True)
    X = data[["Temperature", "Pressure"]].to_numpy(dtype=float)
    y = data[["Density"]].to_numpy(dtype=float)
    return data, X, y


def normalize_data(X):
    X = np.asarray(X, dtype=float)
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    spans = np.where(maxs - mins == 0.0, 1.0, maxs - mins)
    nX = (X - mins) / spans * 0.9 + 0.1
    return nX, mins, maxs


def denormalize_data(X, mins, maxs):
    X = np.asarray(X, dtype=float)
    spans = np.where(maxs - mins == 0.0, 1.0, maxs - mins)
    dX = (X - 0.1) / 0.9 * spans + mins
    return dX


def count_parameters(n_inputs, n_hidden, n_outputs):
    return (n_inputs + 1) * n_hidden + (n_hidden + 1) * n_outputs


def choose_hidden_neurons(n_samples, n_inputs, n_outputs):
    best = 1
    for n_hidden in range(1, 100):
        n_params = count_parameters(n_inputs, n_hidden, n_outputs)
        if n_params < n_samples:
            best = n_hidden
        else:
            break
    return best


def get_model(n_inputs, n_outputs, n_hidden):
    model = Sequential()
    model.add(Input(shape=(n_inputs,)))
    model.add(Dense(n_hidden, activation="sigmoid"))
    model.add(Dense(n_outputs, activation="linear"))
    opt = optimizers.Adam(learning_rate=0.005)
    model.compile(loss="mae", metrics=["mape"], optimizer=opt)
    return model


def evaluate_model(X, y, n_hidden, epochs=1500):
    n_inputs, n_outputs = X.shape[1], y.shape[1]
    cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=22527)
    fold_rows = []
    best_score = np.inf
    best_seed = 22527
    fold_id = 0

    for train_ix, test_ix in cv.split(X):
        fold_id += 1
        X_train, X_test = X[train_ix], X[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]
        seed = 22527 + fold_id
        set_seed(seed)
        model = get_model(n_inputs, n_outputs, n_hidden)
        history = model.fit(X_train, y_train, verbose=0, epochs=epochs, batch_size=len(X_train))
        train_loss, train_mape = model.evaluate(X_train, y_train, verbose=0)
        test_loss, test_mape = model.evaluate(X_test, y_test, verbose=0)
        total_loss, total_mape = model.evaluate(X, y, verbose=0)

        fold_rows.append({
            "fold": fold_id,
            "mae_train": float(train_loss),
            "mae_test": float(test_loss),
            "mape_train": float(train_mape),
            "mape_test": float(test_mape),
            "mae_total": float(total_loss),
            "mape_total": float(total_mape),
            "epochs": len(history.history["loss"]),
            "seed": seed
        })

        if total_mape < best_score:
            best_score = total_mape
            best_seed = seed

        tf.keras.backend.clear_session()

    results = pd.DataFrame(fold_rows)
    return results, best_seed


def fit_final_model(X, y, n_hidden, seed, epochs=1500):
    set_seed(seed)
    model = get_model(X.shape[1], y.shape[1], n_hidden)
    history = model.fit(X, y, verbose=0, epochs=epochs, batch_size=len(X))
    return model, history


def predict_points(model, points_df, minsX, maxsX, minsy, maxsy):
    X_new = points_df[["Temperature", "Pressure"]].to_numpy(dtype=float)
    X_new_norm, _, _ = normalize_data_with_reference(X_new, minsX, maxsX)
    y_pred_norm = model.predict(X_new_norm, verbose=0)
    y_pred = denormalize_data(y_pred_norm, minsy, maxsy).reshape(-1)
    result = points_df.copy()
    result["Predicted_Density"] = y_pred
    return result


def normalize_data_with_reference(X, mins, maxs):
    X = np.asarray(X, dtype=float)
    spans = np.where(maxs - mins == 0.0, 1.0, maxs - mins)
    nX = (X - mins) / spans * 0.9 + 0.1
    return nX, mins, maxs


def predict_curve(model, temperatures, pressure, minsX, maxsX, minsy, maxsy):
    P_line = np.full_like(temperatures, pressure, dtype=float)
    X_line = np.column_stack((temperatures, P_line))
    X_line_norm, _, _ = normalize_data_with_reference(X_line, minsX, maxsX)
    y_line_norm = model.predict(X_line_norm, verbose=0)
    y_line = denormalize_data(y_line_norm, minsy, maxsy).reshape(-1)
    return y_line


def plot_results(data, prediction_table, model, minsX, maxsX, minsy, maxsy):
    pressures = sorted(set(data["Pressure"].tolist()) | set(prediction_table["Pressure"].tolist()))
    t_min = float(data["Temperature"].min())
    t_max = float(data["Temperature"].max())

    for pressure in pressures:
        plt.figure(figsize=(8, 5))

        exp_data = data[data["Pressure"] == pressure].sort_values("Temperature")
        if not exp_data.empty:
            plt.scatter(exp_data["Temperature"], exp_data["Density"], s=50, label="Experimental")

        T_line = np.linspace(t_min, t_max, 300)
        y_line = predict_curve(model, T_line, pressure, minsX, maxsX, minsy, maxsy)
        plt.plot(T_line, y_line, linewidth=2, label="ML model")

        pred_data = prediction_table[prediction_table["Pressure"] == pressure]
        if not pred_data.empty:
            for idx, (_, row) in enumerate(pred_data.iterrows(), start=1):
                plt.scatter(row["Temperature"], row["Predicted_Density"], s=90, label=f"Prediction {idx}")

        plt.xlabel("Temperature [K]")
        plt.ylabel("Density")
        plt.title(f"Butane gas density at P = {pressure:g} bar")
        plt.grid(True, alpha=0.3)
        handles, labels = plt.gca().get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        plt.legend(uniq.values(), uniq.keys())
        plt.tight_layout()
        plt.savefig(f"butane_density_P_{str(pressure).replace('.', '_')}_bar.png", dpi=300)

    plt.show()


def main():
    data, X, y = get_dataset()
    X_norm, minsX, maxsX = normalize_data(X)
    y_norm, minsy, maxsy = normalize_data(y)

    n_samples = X_norm.shape[0]
    n_inputs = X_norm.shape[1]
    n_outputs = y_norm.shape[1]
    n_hidden = choose_hidden_neurons(n_samples, n_inputs, n_outputs)
    n_params = count_parameters(n_inputs, n_hidden, n_outputs)

    cv_results, best_seed = evaluate_model(X_norm, y_norm, n_hidden)
    final_model, history = fit_final_model(X_norm, y_norm, n_hidden, best_seed)
    final_model.save("butane_density_model.keras")

    prediction_points = pd.DataFrame({
        "Temperature": [450.0, 500.0, 550.0],
        "Pressure": [5.0, 5.0, 5.0]
    })

    prediction_table = predict_points(final_model, prediction_points, minsX, maxsX, minsy, maxsy)
    prediction_table.to_csv("butane_density_predictions.csv", index=False)
    cv_results.to_csv("butane_density_cv_results.csv", index=False)

    print(f"Samples: {n_samples}")
    print(f"Inputs: {n_inputs}")
    print(f"Outputs: {n_outputs}")
    print(f"Hidden neurons by rule: {n_hidden}")
    print(f"Trainable parameters: {n_params}")
    print()
    print("Cross-validation summary:")
    print(
        cv_results[["mae_train", "mae_test", "mape_train", "mape_test", "mae_total", "mape_total"]].mean().to_string())
    print()
    print("Predicted values:")
    print(prediction_table.to_string(index=False))

    plot_results(data, prediction_table, final_model, minsX, maxsX, minsy, maxsy)


if __name__ == "__main__":
    main()
