import argparse
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import yaml

from economic_eval_common import (
    build_daily_log_returns_from_pivoted,
    find_asset_file,
    find_exact_or_stem_csv,
    find_matching_window,
    load_har_dates_from_folder,
    load_prediction_workbook,
    nanmean,
    normalize_date,
    read_csv_rows,
    resolve_models_config,
    resolve_path,
    safe_float,
    summarize_groups,
    to_sigma,
    write_csv,
)


def chi_square_df1_sf(value):
    if value is None or math.isnan(value) or value < 0:
        return float("nan")
    return math.erfc(math.sqrt(value / 2.0))


def chi_square_df2_sf(value):
    if value is None or math.isnan(value) or value < 0:
        return float("nan")
    return math.exp(-value / 2.0)


def safe_log(value):
    return math.log(max(value, 1e-300))


def kupiec_lr_uc(exceptions, alpha):
    n_obs = len(exceptions)
    if n_obs == 0:
        return float("nan"), float("nan")
    n_fail = int(np.sum(exceptions))
    fail_rate = n_fail / float(n_obs)

    term_null = ((n_obs - n_fail) * safe_log(1.0 - alpha)) + (n_fail * safe_log(alpha))
    term_alt = ((n_obs - n_fail) * safe_log(1.0 - fail_rate)) + (n_fail * safe_log(fail_rate))
    lr_uc = -2.0 * (term_null - term_alt)
    return lr_uc, chi_square_df1_sf(lr_uc)


def christoffersen_independence_lr(exceptions):
    if len(exceptions) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")

    n00 = n01 = n10 = n11 = 0
    for previous_value, current_value in zip(exceptions[:-1], exceptions[1:]):
        if previous_value == 0 and current_value == 0:
            n00 += 1
        elif previous_value == 0 and current_value == 1:
            n01 += 1
        elif previous_value == 1 and current_value == 0:
            n10 += 1
        else:
            n11 += 1

    total_0 = n00 + n01
    total_1 = n10 + n11
    total = total_0 + total_1
    if total == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    pi_0 = (n01 / float(total_0)) if total_0 > 0 else 0.0
    pi_1 = (n11 / float(total_1)) if total_1 > 0 else 0.0
    pi = ((n01 + n11) / float(total)) if total > 0 else 0.0

    ll_null = ((n00 + n10) * safe_log(1.0 - pi)) + ((n01 + n11) * safe_log(pi))
    ll_alt = (
        n00 * safe_log(1.0 - pi_0)
        + n01 * safe_log(pi_0)
        + n10 * safe_log(1.0 - pi_1)
        + n11 * safe_log(pi_1)
    )
    lr_ind = -2.0 * (ll_null - ll_alt)
    return lr_ind, chi_square_df1_sf(lr_ind), pi_0, pi_1


class LowFreqVaRRunner(object):
    def __init__(self, config, config_path):
        self.config = config
        self.config_path = config_path
        self.project_root = config_path.parent.parent.resolve()

        self.models_cfg = resolve_models_config(
            config,
            self.project_root,
            "low_frequency_folder",
        )
        self.data_cfg = config["data"]
        self.eval_cfg = config["evaluation"]
        self.var_cfg = config["var"]
        self.output_cfg = config["output"]

        self.stock_har_folder = resolve_path(self.data_cfg["stock_har_folder"], self.project_root)
        self.stock_pivoted_price_folder = resolve_path(
            self.data_cfg["stock_pivoted_price_folder"],
            self.project_root,
        )
        self.output_dir = resolve_path(self.output_cfg["output_dir"], self.project_root)

        self.dataset_labels = self.eval_cfg.get("dataset_labels", ["Test"])
        self.variance_epsilon = float(self.eval_cfg.get("variance_epsilon", 1e-12))
        self.prediction_value_type = self.eval_cfg.get("prediction_value_type", "variance")
        self.prediction_tail_length = int(self.eval_cfg.get("prediction_tail_length", 128))
        self.match_atol = float(self.eval_cfg.get("match_atol", 1e-12))
        self.match_rtol = float(self.eval_cfg.get("match_rtol", 1e-6))
        self.confidence_levels = [float(item) for item in self.var_cfg.get("confidence_levels", [0.01, 0.05])]
        self.mean_mode = self.var_cfg.get("mean_mode", "zero")
        self.rolling_window = int(self.var_cfg.get("rolling_window", 22))
        self.min_history = int(self.var_cfg.get("min_history", 10))

        self.prediction_cache = {}
        self.stock_har_cache = {}
        self.stock_return_cache = {}

    def _load_prediction_workbook(self, workbook_path):
        if workbook_path in self.prediction_cache:
            return self.prediction_cache[workbook_path]
        data = load_prediction_workbook(
            Path(workbook_path),
            self.dataset_labels,
            tail_length=self.prediction_tail_length,
        )
        self.prediction_cache[workbook_path] = data
        return data

    def _load_stock_har(self, stock_name):
        if stock_name in self.stock_har_cache:
            return self.stock_har_cache[stock_name]

        path = find_exact_or_stem_csv(self.stock_har_folder, stock_name)
        rows = read_csv_rows(path)

        dates = []
        values = []
        for row in rows:
            if len(row) < 2:
                continue
            date_value = normalize_date(row[0])
            value = safe_float(row[1])
            if date_value is None or value is None:
                continue
            dates.append(date_value)
            values.append(value)

        series = {
            "path": str(path),
            "dates": dates,
            "values": np.asarray(values, dtype=float),
        }
        self.stock_har_cache[stock_name] = series
        return series

    def _load_stock_return_series(self, stock_name):
        if stock_name in self.stock_return_cache:
            return self.stock_return_cache[stock_name]

        path = find_asset_file(
            self.stock_pivoted_price_folder,
            stock_name,
            extensions=[".csv"],
            required_substrings=["pivoted"],
        )
        _, _, return_by_date = build_daily_log_returns_from_pivoted(path)
        ordered_har_dates = load_har_dates_from_folder(self.stock_har_folder, stock_name)

        aligned_dates = []
        aligned_returns = []
        for date_value in ordered_har_dates:
            aligned_dates.append(date_value)
            if date_value in return_by_date:
                aligned_returns.append(float(return_by_date[date_value]))
            else:
                aligned_returns.append(float("nan"))

        if not aligned_returns or np.isnan(np.asarray(aligned_returns, dtype=float)).all():
            raise ValueError(
                "No usable daily returns built for stock '{0}' from '{1}'.".format(
                    stock_name,
                    path,
                )
            )

        result = {
            "dates": aligned_dates,
            "returns": aligned_returns,
        }
        self.stock_return_cache[stock_name] = result
        return result

    def _expected_mean(self, full_return_values, local_index, start_index):
        if self.mean_mode == "zero":
            return 0.0

        if self.mean_mode != "rolling_mean":
            raise ValueError("Unsupported var.mean_mode: '{0}'.".format(self.mean_mode))

        global_index = start_index + local_index
        left = max(0, global_index - self.rolling_window)
        history = [
            value
            for value in full_return_values[left:global_index]
            if not math.isnan(value)
        ]
        if len(history) < self.min_history:
            return None
        return float(np.mean(np.asarray(history, dtype=float)))

    def _compute_var_rows(
        self,
        model_name,
        stock_name,
        predicted_values,
        actual_values,
        full_return_values,
        realized_returns,
        start_index,
    ):
        rows = []
        actual_array = np.asarray(actual_values, dtype=float)
        realized_variance_array = np.asarray(
            [item * item for item in realized_returns],
            dtype=float,
        )
        actual_match_mae = float(np.mean(np.abs(realized_variance_array - actual_array)))

        for alpha in self.confidence_levels:
            z_alpha = NormalDist().inv_cdf(alpha)
            thresholds = []
            exceedances = []
            used_returns = []
            used_sigmas = []

            for local_index, predicted_value in enumerate(predicted_values):
                realized_return = realized_returns[local_index]
                if math.isnan(realized_return):
                    continue

                mu_t = self._expected_mean(full_return_values, local_index, start_index)
                if mu_t is None:
                    continue

                sigma_t = to_sigma(
                    predicted_value,
                    self.prediction_value_type,
                    self.variance_epsilon,
                )
                threshold = mu_t + (z_alpha * sigma_t)
                exceedance = 1 if realized_return < threshold else 0

                thresholds.append(threshold)
                exceedances.append(exceedance)
                used_returns.append(realized_return)
                used_sigmas.append(sigma_t)

            if len(exceedances) >= 1:
                observed_failure_rate = float(np.mean(exceedances))
                mean_threshold = float(np.mean(thresholds))
                mean_var_positive = float(np.mean([-item for item in thresholds]))
                mean_realized_return = float(np.mean(used_returns))
                mean_pred_sigma = float(np.mean(used_sigmas))
                lr_uc, p_uc = kupiec_lr_uc(exceedances, alpha)
                lr_cc = lr_uc
                p_cc = p_uc
            else:
                observed_failure_rate = float("nan")
                mean_threshold = float("nan")
                mean_var_positive = float("nan")
                mean_realized_return = float("nan")
                mean_pred_sigma = float("nan")
                lr_uc = p_uc = lr_cc = p_cc = float("nan")

            rows.append(
                {
                    "model": model_name,
                    "stock": stock_name,
                    "alpha": alpha,
                    "num_observations": len(exceedances),
                    "window_start_index": start_index,
                    "window_end_index": start_index + len(predicted_values) - 1,
                    "actual_match_mae": actual_match_mae,
                    "mean_pred_sigma": mean_pred_sigma,
                    "mean_realized_return": mean_realized_return,
                    "mean_var_threshold": mean_threshold,
                    "mean_var_positive": mean_var_positive,
                    "expected_failure_rate": alpha,
                    "observed_failure_rate": observed_failure_rate,
                    "num_exceedances": int(np.sum(exceedances)) if exceedances else 0,
                    "kupiec_lr_uc": lr_uc,
                    "kupiec_p_value": p_uc,
                    "conditional_coverage_lr": lr_cc,
                    "conditional_coverage_p_value": p_cc,
                }
            )

        return rows

    def validate_inputs(self):
        print("--- Checking low-frequency VaR inputs")
        if not self.stock_har_folder.exists():
            raise FileNotFoundError(
                "Stock HAR folder not found: '{0}'.".format(self.stock_har_folder)
            )
        if not self.stock_pivoted_price_folder.exists():
            raise FileNotFoundError(
                "Stock pivoted price folder not found: '{0}'.".format(self.stock_pivoted_price_folder)
            )

        for model_name, model_cfg in self.models_cfg.items():
            workbook_path = resolve_path(model_cfg["prediction_workbook"], self.project_root)
            if not workbook_path.exists():
                raise FileNotFoundError(
                    "Prediction workbook for model '{0}' not found: '{1}'.".format(
                        model_name,
                        workbook_path,
                    )
                )
            predictions = self._load_prediction_workbook(str(workbook_path))
            if not predictions:
                raise ValueError(
                    "Model '{0}' has no rows after dataset filtering.".format(model_name)
                )
            print(
                "   model={0}, sheets={1}, workbook={2}".format(
                    model_name,
                    len(predictions),
                    workbook_path,
                )
            )

    def run(self):
        self.validate_inputs()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        var_rows = []
        alignment_rows = []

        for model_name, model_cfg in self.models_cfg.items():
            workbook_path = str(resolve_path(model_cfg["prediction_workbook"], self.project_root))
            predictions = self._load_prediction_workbook(workbook_path)
            print("--- Running low-frequency VaR for model: {0}".format(model_name))

            for stock_name, prediction_data in predictions.items():
                stock_har = self._load_stock_har(stock_name)
                return_series = self._load_stock_return_series(stock_name)
                actual_array = np.asarray(prediction_data["actual"], dtype=float)
                predicted_values = list(prediction_data["predicted"])
                start_index = find_matching_window(
                    stock_har["values"],
                    actual_array,
                    self.match_atol,
                    self.match_rtol,
                )

                full_return_values = return_series["returns"]
                realized_returns = full_return_values[start_index : start_index + len(predicted_values)]
                if len(realized_returns) != len(predicted_values):
                    raise ValueError(
                        "Realized return length mismatch for stock '{0}': got {1}, expected {2}."
                        .format(stock_name, len(realized_returns), len(predicted_values))
                    )

                alignment_rows.append(
                    {
                        "model": model_name,
                        "stock": stock_name,
                        "prediction_workbook": workbook_path,
                        "window_start_index": start_index,
                        "window_end_index": start_index + len(predicted_values) - 1,
                        "window_start_date": stock_har["dates"][start_index],
                        "window_end_date": stock_har["dates"][start_index + len(predicted_values) - 1],
                        "prediction_rows": len(predicted_values),
                        "dataset_labels": "|".join(sorted(set(prediction_data["dataset"]))),
                        "actual_match_mae": float(
                            np.mean(
                                np.abs(
                                    np.asarray([item * item for item in realized_returns], dtype=float)
                                    - actual_array
                                )
                            )
                        ),
                    }
                )

                var_rows.extend(
                    self._compute_var_rows(
                        model_name,
                        stock_name,
                        predicted_values,
                        actual_array,
                        full_return_values,
                        realized_returns,
                        start_index,
                    )
                )

        var_summary = summarize_groups(
            var_rows,
            group_keys=["model", "alpha"],
            numeric_keys=[
                "observed_failure_rate",
                "conditional_coverage_p_value",
            ],
        )

        write_csv(self.output_dir / "var_detail.csv", var_rows)
        write_csv(self.output_dir / "var_summary.csv", var_summary)
        if self.output_cfg.get("write_alignment_detail", True):
            write_csv(self.output_dir / "alignment_detail.csv", alignment_rows)

        print("--- Finished low-frequency VaR")


def load_config(config_path):
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    required_sections = [
        "data",
        "evaluation",
        "var",
        "output",
    ]
    for section in required_sections:
        if section not in config:
            raise ValueError("Config is missing section '{0}'.".format(section))
    if not config.get("models") and not config.get("prediction_sources"):
        raise ValueError("Config must provide either 'models' or 'prediction_sources'.")
    return config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run low-frequency daily VaR backtests."
    )
    parser.add_argument(
        "--config",
        default="config/low_freq_var.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only validate inputs and config, do not run the tests.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    runner = LowFreqVaRRunner(config, config_path)
    if args.check_only:
        runner.validate_inputs()
        print("--- Check completed")
        return
    runner.run()


if __name__ == "__main__":
    main()
