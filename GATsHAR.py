import os
import traceback
from datetime import datetime

import optuna
import pandas as pd
import yaml

from train_single_final import train


DEFAULT_CONFIG_PATH = os.environ.get("GATHAR_CONFIG_PATH", "config/GNN_param_optuna.yaml")


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def suggest_param(trial: optuna.trial.Trial, param: str, values, dtype: str):
    if param == "hidden_layout":
        str_choices = [str(layout) for layout in values]
        suggested_str = trial.suggest_categorical(param, str_choices)
        return eval(suggested_str)

    if dtype == "cat":
        return trial.suggest_categorical(param, values)
    if dtype == "float":
        min_val, max_val = values
        return trial.suggest_float(param, min_val, max_val, log=(param == "learning_rate"))
    if dtype == "int":
        min_val, max_val = values
        return trial.suggest_int(param, min_val, max_val)

    raise ValueError(f"Unsupported hyperparameter dtype: {dtype}")


def objective(trial: optuna.trial.Trial) -> float:
    config_path = trial.study.user_attrs["config_path"]
    p = load_config(config_path)

    for param in p.get("grid", []):
        if param not in p.get("hyperparameters", {}):
            continue
        values, dtype = p["hyperparameters"][param]
        p[param] = suggest_param(trial, param, values, dtype)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    trial_folder = os.path.join(
        trial.study.user_attrs["main_output_folder"],
        f"trial_{trial.number}_{timestamp}",
    )
    os.makedirs(trial_folder, exist_ok=True)

    try:
        metrics = train(p=p, trial_folder=trial_folder)
        for key, value in metrics.items():
            trial.set_user_attr(key, value)
        trial.set_user_attr("output_folder", trial_folder)
        return metrics.get("min_validation_loss", float("inf"))
    except Exception as exc:
        print(f"Trial #{trial.number} failed: {exc}")
        print("\n" + "=" * 25 + " Traceback " + "=" * 25)
        traceback.print_exc()
        print("=" * 65 + "\n")
        return float("inf")


def main():
    config_path = DEFAULT_CONFIG_PATH
    p = load_config(config_path)

    main_output_folder = os.path.join("output", f"{p['modelname']}_tuning")
    os.makedirs(main_output_folder, exist_ok=True)

    storage_name = f"sqlite:///{main_output_folder}/optuna_study.db"
    study = optuna.create_study(
        study_name=p["modelname"],
        direction="minimize",
        storage=storage_name,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=p["seed"]),
    )
    study.set_user_attr("config_path", config_path)
    study.set_user_attr("main_output_folder", main_output_folder)

    study.optimize(objective, n_trials=p["n_trials"])

    print("\n--- Hyperparameter tuning finished ---")
    results = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE or trial.value is None:
            continue

        params = dict(trial.params)
        if "hidden_layout" in params:
            params["hidden_layout"] = str(eval(params["hidden_layout"]))

        row = {
            **params,
            **trial.user_attrs,
            "trial_number": trial.number,
            "value (best_val_loss)": trial.value,
        }
        results.append(row)

    results_df = pd.DataFrame(results)
    metric_cols = [
        "trial_number",
        "value (best_val_loss)",
        "best_epoch",
        "validation_rmse_h",
        "validation_mse_h",
        "validation_qlike_h",
        "validation_rmse_l",
        "validation_mse_l",
        "validation_qlike_l",
        "test_rmse_h",
        "test_mse_h",
        "test_qlike_h",
        "test_rmse_l",
        "test_mse_l",
        "test_qlike_l",
        "output_folder",
    ]
    param_cols = [col for col in p.get("grid", []) if col in results_df.columns]
    ordered_cols = [col for col in metric_cols + param_cols if col in results_df.columns]
    results_df = results_df[ordered_cols].sort_values(
        by="value (best_val_loss)", ascending=True
    )

    summary_path = os.path.join(main_output_folder, "tuning_summary.csv")
    results_df.to_csv(summary_path, index=False, float_format="%.12f")

    best_params_display = dict(study.best_params)
    if "hidden_layout" in best_params_display:
        best_params_display["hidden_layout"] = eval(best_params_display["hidden_layout"])

    print("Best params:")
    print(best_params_display)
    print(f"Best validation objective: {study.best_trial.value}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
