import argparse
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import yaml

from economic_eval_common import (
    find_asset_file,
    load_har_dates_from_folder,
    load_prediction_workbook,
    nanmean,
    normalize_key,
    normalize_date,
    normalize_time_label,
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


class HighFreqVaRRunner(object):
    def __init__(self, config, config_path):
        self.config = config
        self.config_path = config_path
        self.project_root = config_path.parent.parent.resolve()

        self.models_cfg = resolve_models_config(
            config,
            self.project_root,
            "high_frequency_folder",
        )
        self.data_cfg = config["data"]
        self.eval_cfg = config["evaluation"]
        self.var_cfg = config["var"]
        self.output_cfg = config["output"]

        self.stock_return_folder = resolve_path(
            self.data_cfg["high_freq_stock_return_folder"],
            self.project_root,
        )
        self.stock_pivoted_price_folder = None
        self.stock_har_folder = None
        if self.data_cfg.get("stock_pivoted_price_folder"):
            self.stock_pivoted_price_folder = resolve_path(
                self.data_cfg["stock_pivoted_price_folder"],
                self.project_root,
            )
        if self.data_cfg.get("stock_har_folder"):
            self.stock_har_folder = resolve_path(
                self.data_cfg["stock_har_folder"],
                self.project_root,
            )
        self.generated_return_output_dir = None
        if self.data_cfg.get("generated_return_output_dir"):
            self.generated_return_output_dir = resolve_path(
                self.data_cfg["generated_return_output_dir"],
                self.project_root,
            )
        self.output_dir = resolve_path(self.output_cfg["output_dir"], self.project_root)

        self.dataset_labels = self.eval_cfg.get("dataset_labels", ["Test"])
        self.variance_epsilon = float(self.eval_cfg.get("variance_epsilon", 1e-12))
        self.prediction_value_type = self.eval_cfg.get("prediction_value_type", "variance")
        self.prediction_tail_length = int(self.eval_cfg.get("prediction_tail_length", 128))
        self.confidence_levels = [float(item) for item in self.var_cfg.get("confidence_levels", [0.01, 0.05])]
        self.mean_mode = self.var_cfg.get("mean_mode", "zero")
        self.rolling_window = int(self.var_cfg.get("rolling_window", 60))
        self.min_history = int(self.var_cfg.get("min_history", 20))
        self.return_file_type = self.var_cfg.get("return_file_type", "return")
        self.seq_length = int(self.eval_cfg.get("seq_length", 15))
        self.intraday_points = int(self.eval_cfg.get("intraday_points", 3))
        self.train_proportion = float(self.eval_cfg.get("train_proportion", 0.7))
        self.validation_proportion = float(self.eval_cfg.get("validation_proportion", 0.1))
        self.robustness_proportion = float(self.eval_cfg.get("robustness_proportion", 0.7))

        self.prediction_cache = {}
        self.stock_return_cache = {}
        self.high_vol_cache = {}

    def _build_high_freq_series_from_original_prices(self, stock_name):
        if self.stock_pivoted_price_folder is None:
            raise FileNotFoundError(
                "Config is missing data.stock_pivoted_price_folder for original high-frequency prices."
            )
        if self.stock_har_folder is None:
            raise FileNotFoundError(
                "Config is missing data.stock_har_folder for date alignment."
            )

        path = find_asset_file(
            self.stock_pivoted_price_folder,
            stock_name,
            extensions=[".csv"],
            required_substrings=["pivoted"],
        )
        har_dates = load_har_dates_from_folder(self.stock_har_folder, stock_name)
        rows = read_csv_rows(path)
        if not rows or len(rows[0]) < 2:
            raise ValueError("Pivoted minute file is empty or malformed: '{0}'.".format(path))

        all_dates = [normalize_date(item) for item in rows[0][1:]]
        time_rows = {}
        for row in rows[1:]:
            if not row:
                continue
            time_label = normalize_time_label(row[0])
            if time_label not in {"10:15", "11:30", "15:00"}:
                continue
            values = list(row[1:])
            if len(values) < len(all_dates):
                values.extend([None] * (len(all_dates) - len(values)))
            time_rows[time_label] = values

        required_times = ["10:15", "11:30", "15:00"]
        missing_times = [item for item in required_times if item not in time_rows]
        if missing_times:
            raise ValueError(
                "Pivoted minute file '{0}' is missing required times: {1}".format(
                    path,
                    ", ".join(missing_times),
                )
            )

        date_to_index = {}
        for idx, date_value in enumerate(all_dates):
            if date_value is not None and date_value not in date_to_index:
                date_to_index[date_value] = idx

        selected_pairs = []
        missing_dates = []
        for date_value in har_dates:
            if date_value not in date_to_index:
                missing_dates.append(date_value)
                continue
            selected_pairs.append((date_to_index[date_value], date_value))

        if missing_dates:
            raise ValueError(
                "Pivoted minute file '{0}' is missing {1} required dates. First missing date: {2}".format(
                    path,
                    len(missing_dates),
                    missing_dates[0],
                )
            )

        returns_by_date = {}
        close_by_date = {}
        flat_returns = []
        selected_dates = []
        previous_close = None

        for date_idx, date_value in selected_pairs:
            open_1015 = safe_float(time_rows["10:15"][date_idx])
            mid_1130 = safe_float(time_rows["11:30"][date_idx])
            close_1500 = safe_float(time_rows["15:00"][date_idx])

            if (
                open_1015 is None
                or mid_1130 is None
                or close_1500 is None
                or open_1015 <= 0.0
                or mid_1130 <= 0.0
                or close_1500 <= 0.0
            ):
                raise ValueError(
                    "Pivoted minute file '{0}' does not have a usable price set for date '{1}'.".format(
                        path,
                        date_value,
                    )
                )

            if previous_close is None or previous_close <= 0.0:
                first_return = float("nan")
            else:
                first_return = math.log(open_1015) - math.log(previous_close)
            second_return = math.log(mid_1130) - math.log(open_1015)
            third_return = math.log(close_1500) - math.log(mid_1130)

            day_returns = [first_return, second_return, third_return]
            returns_by_date[date_value] = day_returns
            close_by_date[date_value] = float(close_1500)
            selected_dates.append(date_value)
            flat_returns.extend(day_returns)
            previous_close = float(close_1500)

        return_values = [float(item) for item in flat_returns]
        vol_values = [float(item * item) for item in return_values]

        if self.generated_return_output_dir is not None:
            self.generated_return_output_dir.mkdir(parents=True, exist_ok=True)
            rows = []
            for date_value in selected_dates:
                day_returns = returns_by_date[date_value]
                rows.append(
                    {
                        "date": date_value,
                        "segment_1_return": day_returns[0],
                        "segment_2_return": day_returns[1],
                        "segment_3_return": day_returns[2],
                        "segment_1_variance": day_returns[0] * day_returns[0],
                        "segment_2_variance": day_returns[1] * day_returns[1],
                        "segment_3_variance": day_returns[2] * day_returns[2],
                    }
                )
            write_csv(self.generated_return_output_dir / (stock_name + "_high_freq_segments.csv"), rows)

        return {
            "dates": selected_dates,
            "returns": return_values,
            "variances": vol_values,
        }

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

    def _load_high_freq_return_values(self, stock_name):
        if stock_name in self.stock_return_cache:
            return self.stock_return_cache[stock_name]

        if self.stock_pivoted_price_folder is not None:
            series = self._build_high_freq_series_from_original_prices(stock_name)
            self.stock_return_cache[stock_name] = series["returns"]
            self.high_vol_cache[stock_name] = series["variances"]
            return series["returns"]

        path = find_exact_or_stem_csv(self.stock_return_folder, stock_name)
        rows = read_csv_rows(path)
        if not rows:
            raise ValueError("High-frequency return file is empty: '{0}'.".format(path))

        values = []
        start_row = 0
        if len(rows[0]) >= 2 and safe_float(rows[0][-1]) is None:
            start_row = 1

        for row in rows[start_row:]:
            if not row:
                continue
            if len(row) == 1:
                value = safe_float(row[0])
            else:
                value = safe_float(row[-1])
            if value is None:
                continue
            values.append(float(value))

        if not values:
            raise ValueError("No usable numeric values found in '{0}'.".format(path))

        if self.return_file_type == "close":
            if len(values) < 2:
                raise ValueError("Close series '{0}' is too short.".format(path))
            returns = []
            for previous_close, current_close in zip(values[:-1], values[1:]):
                if previous_close == 0.0:
                    returns.append(float("nan"))
                else:
                    returns.append((current_close / previous_close) - 1.0)
            values = returns

        self.stock_return_cache[stock_name] = values
        return values

    def _get_high_freq_variance_values(self, stock_name):
        if stock_name in self.high_vol_cache:
            return self.high_vol_cache[stock_name]
        self._load_high_freq_return_values(stock_name)
        return self.high_vol_cache[stock_name]

    def _resolve_evaluation_window(self, total_steps, prediction_length):
        sample_count = total_steps - self.seq_length - self.intraday_points
        if sample_count <= 0:
            raise ValueError("Not enough high-frequency observations to build the dataset.")
        if sample_count % self.intraday_points != 0:
            raise ValueError(
                "High-frequency sample count {0} is not divisible by intraday_points={1}."
                .format(sample_count, self.intraday_points)
            )

        total_days = sample_count // self.intraday_points
        train_days = int(self.train_proportion * total_days)
        validation_days = int(self.validation_proportion * total_days)
        train_steps = train_days * self.intraday_points
        validation_steps = validation_days * self.intraday_points

        split_ranges = {
            "validation": (
                train_steps,
                train_steps + validation_steps,
            ),
            "test": (
                train_steps + validation_steps,
                sample_count,
            ),
        }

        normalized_labels = [str(label).strip().lower() for label in self.dataset_labels]
        supported_labels = {"validation", "test"}
        unknown_labels = [
            label for label in normalized_labels if label and label not in supported_labels
        ]
        if unknown_labels:
            raise ValueError(
                "Unsupported high-frequency dataset_labels for alignment: {0}."
                .format(", ".join(sorted(set(unknown_labels))))
            )

        selected_labels = [label for label in ["validation", "test"] if label in normalized_labels]
        if not selected_labels:
            raise ValueError("No supported dataset labels were selected for high-frequency alignment.")

        selected_start = split_ranges[selected_labels[0]][0]
        selected_end = split_ranges[selected_labels[-1]][1]
        selected_length = selected_end - selected_start
        if prediction_length > selected_length:
            raise ValueError(
                "Prediction length {0} exceeds selected high-frequency window length {1}."
                .format(prediction_length, selected_length)
            )

        sample_window_start = selected_end - prediction_length
        sample_window_end = sample_window_start + prediction_length - 1
        raw_window_start = sample_window_start + self.seq_length
        raw_window_end = raw_window_start + prediction_length - 1

        if raw_window_end >= total_steps:
            raise ValueError(
                "Resolved high-frequency raw window [{0}, {1}] exceeds total steps {2}."
                .format(raw_window_start, raw_window_end, total_steps)
            )

        return {
            "sample_count": sample_count,
            "total_days": total_days,
            "train_days": train_days,
            "validation_days": validation_days,
            "selected_labels": "|".join(selected_labels),
            "selected_start_sample_index": selected_start,
            "selected_end_sample_index": selected_end - 1,
            "window_start_sample_index": sample_window_start,
            "window_end_sample_index": sample_window_end,
            "window_start_index": raw_window_start,
            "window_end_index": raw_window_end,
        }

    def _resolve_model_evaluation_window(self, model_name, total_steps, prediction_length):
        if str(model_name).strip().upper() == "GATHAR":
            return self._resolve_gathar_gat2xlsx_window(total_steps, prediction_length)
        return self._resolve_evaluation_window(total_steps, prediction_length)

    def _resolve_gathar_gat2xlsx_window(self, total_steps, prediction_length):
        sample_count = total_steps - self.seq_length - self.intraday_points
        if sample_count <= 0:
            raise ValueError("Not enough high-frequency observations to build the GATHAR dataset.")
        if sample_count % self.intraday_points != 0:
            raise ValueError(
                "GATHAR high-frequency sample count {0} is not divisible by intraday_points={1}."
                .format(sample_count, self.intraday_points)
            )

        total_days = sample_count // self.intraday_points
        robustness_limit_days = int(self.robustness_proportion * total_days)
        truncated_sample_count = robustness_limit_days * self.intraday_points
        if truncated_sample_count <= 0:
            raise ValueError("Not enough high-frequency observations after GATHAR robustness truncation.")

        train_days = int(self.train_proportion * robustness_limit_days)
        validation_days = int(self.validation_proportion * robustness_limit_days)
        train_steps = train_days * self.intraday_points
        validation_steps = validation_days * self.intraday_points

        split_ranges = {
            "validation": (
                train_steps,
                train_steps + validation_steps,
            ),
            "test": (
                train_steps + validation_steps,
                truncated_sample_count,
            ),
        }

        normalized_labels = [str(label).strip().lower() for label in self.dataset_labels]
        supported_labels = {"validation", "test"}
        unknown_labels = [
            label for label in normalized_labels if label and label not in supported_labels
        ]
        if unknown_labels:
            raise ValueError(
                "Unsupported high-frequency dataset_labels for GATHAR alignment: {0}."
                .format(", ".join(sorted(set(unknown_labels))))
            )

        selected_labels = [label for label in ["validation", "test"] if label in normalized_labels]
        if not selected_labels:
            raise ValueError("No supported dataset labels were selected for GATHAR high-frequency alignment.")

        selected_start = split_ranges[selected_labels[0]][0]
        selected_end = split_ranges[selected_labels[-1]][1]
        selected_length = selected_end - selected_start
        if prediction_length > selected_length:
            raise ValueError(
                "Prediction length {0} exceeds selected GATHAR window length {1}."
                .format(prediction_length, selected_length)
            )

        sample_window_start = selected_end - prediction_length
        sample_window_end = sample_window_start + prediction_length - 1
        raw_window_start = sample_window_start + self.seq_length
        raw_window_end = raw_window_start + prediction_length - 1

        if raw_window_end >= total_steps:
            raise ValueError(
                "Resolved GATHAR raw window [{0}, {1}] exceeds total steps {2}."
                .format(raw_window_start, raw_window_end, total_steps)
            )

        return {
            "sample_count": sample_count,
            "total_days": total_days,
            "robustness_limit_days": robustness_limit_days,
            "robustness_limit_steps": truncated_sample_count,
            "train_days": train_days,
            "validation_days": validation_days,
            "selected_labels": "|".join(selected_labels),
            "selected_start_sample_index": selected_start,
            "selected_end_sample_index": selected_end - 1,
            "window_start_sample_index": sample_window_start,
            "window_end_sample_index": sample_window_end,
            "window_start_index": raw_window_start,
            "window_end_index": raw_window_end,
            "alignment_logic": "gat2xlsx_gathar",
        }

    def _slice_realized_returns(self, stock_name, start_index, window_length):
        full_values = self._load_high_freq_return_values(stock_name)
        if len(full_values) >= start_index + window_length:
            return np.asarray(full_values[start_index : start_index + window_length], dtype=float), "full_series"
        if len(full_values) == window_length:
            return np.asarray(full_values, dtype=float), "evaluation_window_only"

        raise ValueError(
            "High-frequency return file for '{0}' has length {1}, expected {2} "
            "for full-series alignment or {3} for evaluation-window-only alignment."
            .format(stock_name, len(full_values), start_index + window_length, window_length)
        )

    def _expected_mean(self, full_return_values, local_index, start_index, alignment_mode):
        if self.mean_mode == "zero":
            return 0.0

        if self.mean_mode != "rolling_mean":
            raise ValueError("Unsupported var.mean_mode: '{0}'.".format(self.mean_mode))

        if alignment_mode == "full_series":
            global_index = start_index + local_index
            left = max(0, global_index - self.rolling_window)
            history = [
                value
                for value in full_return_values[left:global_index]
                if not math.isnan(value)
            ]
        else:
            left = max(0, local_index - self.rolling_window)
            history = [
                value
                for value in full_return_values[left:local_index]
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
        realized_returns,
        start_index,
        alignment_mode,
    ):
        full_return_values = self._load_high_freq_return_values(stock_name)
        full_variance_values = self._get_high_freq_variance_values(stock_name)
        rows = []

        actual_array = np.asarray(actual_values, dtype=float)
        realized_variance_array = np.asarray(
            full_variance_values[start_index : start_index + len(predicted_values)],
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

                mu_t = self._expected_mean(
                    full_return_values,
                    local_index,
                    start_index,
                    alignment_mode,
                )
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
                    "alignment_mode": alignment_mode,
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
        print("--- Checking high-frequency VaR inputs")
        if self.stock_pivoted_price_folder is not None:
            if not self.stock_pivoted_price_folder.exists():
                raise FileNotFoundError(
                    "Stock pivoted price folder not found: '{0}'.".format(
                        self.stock_pivoted_price_folder,
                    )
                )
            if self.stock_har_folder is None or not self.stock_har_folder.exists():
                raise FileNotFoundError(
                    "Stock HAR folder not found: '{0}'.".format(self.stock_har_folder)
                )
        elif not self.stock_return_folder.exists():
            raise FileNotFoundError(
                "High-frequency return folder not found: '{0}'.".format(self.stock_return_folder)
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
            first_stock_name = sorted(predictions.keys(), key=normalize_key)[0]
            full_variance_values = self._get_high_freq_variance_values(first_stock_name)
            print("   derived_high_freq_steps={0}".format(len(full_variance_values)))

    def run(self):
        self.validate_inputs()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        var_rows = []
        alignment_rows = []

        for model_name, model_cfg in self.models_cfg.items():
            workbook_path = str(resolve_path(model_cfg["prediction_workbook"], self.project_root))
            predictions = self._load_prediction_workbook(workbook_path)
            print("--- Running high-frequency VaR for model: {0}".format(model_name))

            for stock_name, prediction_data in predictions.items():
                predicted_values = list(prediction_data["predicted"])
                actual_values = list(prediction_data["actual"])
                total_steps = len(self._get_high_freq_variance_values(stock_name))
                window_info = self._resolve_model_evaluation_window(
                    model_name,
                    total_steps,
                    len(predicted_values),
                )
                start_index = window_info["window_start_index"]
                realized_returns, alignment_mode = self._slice_realized_returns(
                    stock_name,
                    start_index,
                    len(predicted_values),
                )

                alignment_rows.append(
                    {
                        "model": model_name,
                        "stock": stock_name,
                        "prediction_workbook": workbook_path,
                        "sample_window_start_index": window_info["window_start_sample_index"],
                        "sample_window_end_index": window_info["window_end_sample_index"],
                        "window_start_index": start_index,
                        "window_end_index": start_index + len(predicted_values) - 1,
                        "prediction_rows": len(predicted_values),
                        "dataset_labels": "|".join(sorted(set(prediction_data["dataset"]))),
                        "aligned_dataset_labels": window_info["selected_labels"],
                        "alignment_mode": alignment_mode,
                        "actual_match_mae": float(
                            np.mean(
                                np.abs(
                                    np.asarray(
                                        self._get_high_freq_variance_values(stock_name)[
                                            start_index : start_index + len(predicted_values)
                                        ],
                                        dtype=float,
                                    )
                                    - np.asarray(actual_values, dtype=float)
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
                        actual_values,
                        realized_returns,
                        start_index,
                        alignment_mode,
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

        print("--- Finished high-frequency VaR")


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
        description="Run high-frequency intraday VaR backtests."
    )
    parser.add_argument(
        "--config",
        default="config/high_freq_var.yaml",
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
    runner = HighFreqVaRRunner(config, config_path)
    if args.check_only:
        runner.validate_inputs()
        print("--- Check completed")
        return
    runner.run()


if __name__ == "__main__":
    main()
