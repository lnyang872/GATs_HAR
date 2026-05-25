import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from economic_eval_common import (
    build_daily_log_returns_from_pivoted,
    build_scaled_daily_realized_variance_from_pivoted,
    find_exact_or_stem_csv,
    find_asset_file,
    find_matching_window,
    load_daily_close_series_from_csv,
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


class LowFreqHedgingRunner(object):
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
        self.pairing_cfg = config["pairing"]
        self.hedging_cfg = config["hedging"]
        self.output_cfg = config["output"]

        self.stock_har_folder = resolve_path(self.data_cfg["stock_har_folder"], self.project_root)
        self.energy_har_folder = resolve_path(self.data_cfg["energy_har_folder"], self.project_root)
        self.corr_folder = resolve_path(self.data_cfg["stock_energy_corr_folder"], self.project_root)
        self.stock_return_folder = resolve_path(self.data_cfg["stock_return_folder"], self.project_root)
        self.future_return_folder = resolve_path(self.data_cfg["future_return_folder"], self.project_root)
        self.output_dir = resolve_path(self.output_cfg["output_dir"], self.project_root)
        self.stock_pivoted_price_folder = None
        self.future_daily_price_folder = None
        self.future_pivoted_price_folder = None
        self.generated_return_output_dir = None
        if self.data_cfg.get("stock_pivoted_price_folder"):
            self.stock_pivoted_price_folder = resolve_path(
                self.data_cfg["stock_pivoted_price_folder"],
                self.project_root,
            )
        if self.data_cfg.get("future_daily_price_folder"):
            self.future_daily_price_folder = resolve_path(
                self.data_cfg["future_daily_price_folder"],
                self.project_root,
            )
        if self.data_cfg.get("future_pivoted_price_folder"):
            self.future_pivoted_price_folder = resolve_path(
                self.data_cfg["future_pivoted_price_folder"],
                self.project_root,
            )
        if self.data_cfg.get("generated_return_output_dir"):
            self.generated_return_output_dir = resolve_path(
                self.data_cfg["generated_return_output_dir"],
                self.project_root,
            )

        self.match_atol = float(self.eval_cfg.get("match_atol", 1e-12))
        self.match_rtol = float(self.eval_cfg.get("match_rtol", 1e-6))
        self.variance_epsilon = float(self.eval_cfg.get("variance_epsilon", 1e-12))
        self.prediction_value_type = self.eval_cfg.get("prediction_value_type", "variance")
        self.future_har_value_type = self.eval_cfg.get("future_har_value_type", "variance")
        self.prediction_tail_length = int(self.eval_cfg.get("prediction_tail_length", 128))
        self.dataset_labels = self.eval_cfg.get("dataset_labels", ["Test"])
        self.corr_mode = self.pairing_cfg.get("corr_mode", "rolling_return")
        self.corr_window = int(self.pairing_cfg.get("corr_window", 22))
        self.sigma_window = int(self.hedging_cfg.get("sigma_window", self.corr_window))

        self.stock_har_cache = {}
        self.future_har_cache = {}
        self.corr_cache = {}
        self.stock_return_cache = {}
        self.future_return_cache = {}
        self.stock_ordered_dates_cache = {}
        self.prediction_cache = {}
        self.future_names = sorted(path.stem for path in self.energy_har_folder.glob("*.csv"))

    def _load_har_series(self, asset_name, folder, cache):
        if asset_name in cache:
            return cache[asset_name]

        path = find_exact_or_stem_csv(folder, asset_name)
        rows = read_csv_rows(path)
        dates = []
        values = []
        for row in rows:
            if not row or len(row) < 2:
                continue
            date_value = normalize_date(row[0])
            series_value = safe_float(row[1])
            if date_value is None or series_value is None:
                continue
            dates.append(date_value)
            values.append(series_value)

        if not dates:
            raise ValueError("No usable HAR data found in '{0}'.".format(path))

        series = {
            "path": str(path),
            "dates": dates,
            "values": np.asarray(values, dtype=float),
            "date_to_value": {dates[idx]: values[idx] for idx in range(len(dates))},
        }
        cache[asset_name] = series
        return series

    def _load_stock_har(self, stock_name):
        return self._load_har_series(stock_name, self.stock_har_folder, self.stock_har_cache)

    def _load_future_har(self, future_name):
        if self.future_pivoted_price_folder is not None and self.future_pivoted_price_folder.exists():
            if future_name in self.future_har_cache:
                return self.future_har_cache[future_name]

            path = find_asset_file(
                self.future_pivoted_price_folder,
                future_name,
                extensions=[".csv"],
                required_substrings=["pivoted"],
            )
            ordered_har_dates = load_har_dates_from_folder(self.energy_har_folder, future_name)
            rv_data = build_scaled_daily_realized_variance_from_pivoted(path, ordered_har_dates)
            dates = list(rv_data["dates"])
            values = [rv_data["realized_variance_by_date"][date_key] for date_key in dates]
            series = {
                "path": str(path),
                "dates": dates,
                "values": np.asarray(values, dtype=float),
                "date_to_value": {dates[idx]: values[idx] for idx in range(len(dates))},
            }
            self.future_har_cache[future_name] = series
            return series

        return self._load_har_series(future_name, self.energy_har_folder, self.future_har_cache)

    def _load_corr_series(self, future_name, stock_name):
        cache_key = future_name + "||" + stock_name
        if cache_key in self.corr_cache:
            return self.corr_cache[cache_key]

        if self.corr_mode == "rolling_return":
            stock_har = self._load_stock_har(stock_name)
            stock_returns = self._load_stock_returns(stock_name)
            future_returns = self._load_future_returns(future_name)
            corr_values = self._build_corr_series_from_returns(
                stock_har["dates"],
                stock_returns,
                future_returns,
            )
            self.corr_cache[cache_key] = corr_values
            return corr_values

        path = find_exact_or_stem_csv(self.corr_folder, future_name + "_" + stock_name)
        rows = read_csv_rows(path)
        values = []
        for row in rows:
            if not row:
                continue
            value = safe_float(row[0])
            if value is None:
                continue
            values.append(value)

        if not values:
            raise ValueError("No usable correlation data found in '{0}'.".format(path))

        self.corr_cache[cache_key] = values
        return values

    def _build_corr_series_from_returns(self, ordered_dates, stock_returns, future_returns):
        values = []
        aligned_stock = []
        aligned_future = []

        for current_date in ordered_dates:
            stock_value = stock_returns.get(current_date)
            future_value = future_returns.get(current_date)
            if stock_value is None or future_value is None:
                values.append(float("nan"))
                continue

            aligned_stock.append(float(stock_value))
            aligned_future.append(float(future_value))

            left = max(0, len(aligned_stock) - self.corr_window)
            window_stock = np.asarray(aligned_stock[left:], dtype=float)
            window_future = np.asarray(aligned_future[left:], dtype=float)

            if (
                window_stock.size < 2
                or np.std(window_stock) <= 0.0
                or np.std(window_future) <= 0.0
            ):
                values.append(float("nan"))
                continue

            values.append(float(np.corrcoef(window_stock, window_future)[0, 1]))

        if not values:
            raise ValueError("No usable correlation values were built from return series.")
        return values

    def _build_rolling_sigma_series_from_returns(self, ordered_dates, return_by_date, window_size):
        values = []
        aligned = []

        for current_date in ordered_dates:
            return_value = return_by_date.get(current_date)
            if return_value is None or np.isnan(return_value):
                values.append(float("nan"))
                continue

            aligned.append(float(return_value))
            left = max(0, len(aligned) - window_size)
            window = np.asarray(aligned[left:], dtype=float)
            if window.size < 2 or np.std(window, ddof=1) <= 0.0:
                values.append(float("nan"))
                continue
            values.append(float(np.std(window, ddof=1)))

        return values

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

    def _load_return_series(self, folder, asset_name, cache):
        if asset_name in cache:
            return cache[asset_name]

        path = find_exact_or_stem_csv(folder, asset_name)
        rows = read_csv_rows(path)
        if not rows:
            raise ValueError("Return file is empty: '{0}'.".format(path))

        date_to_value = {}
        start_row = 0
        date_idx = 0
        value_idx = 1
        if rows and len(rows[0]) >= 2 and safe_float(rows[0][1]) is None:
            start_row = 1

        for row in rows[start_row:]:
            if not row or len(row) < 2:
                continue
            date_value = normalize_date(row[date_idx])
            return_value = safe_float(row[value_idx])
            if date_value is None or return_value is None:
                continue
            date_to_value[date_value] = return_value

        if not date_to_value:
            raise ValueError("No usable return data found in '{0}'.".format(path))

        cache[asset_name] = date_to_value
        return date_to_value

    def _load_stock_returns(self, stock_name):
        if self.stock_pivoted_price_folder is not None and self.stock_pivoted_price_folder.exists():
            ordered_har_dates = load_har_dates_from_folder(self.stock_har_folder, stock_name)
            path = find_asset_file(
                self.stock_pivoted_price_folder,
                stock_name,
                extensions=[".csv"],
                required_substrings=["pivoted"],
            )
            ordered_dates, _, return_by_date = build_daily_log_returns_from_pivoted(path)
            filtered = {date_key: return_by_date[date_key] for date_key in ordered_har_dates if date_key in return_by_date}
            if not filtered:
                raise ValueError(
                    "No overlapping stock daily returns were built for '{0}' from '{1}'.".format(
                        stock_name,
                        path,
                    )
                )
            self.stock_return_cache[stock_name] = filtered
            self.stock_ordered_dates_cache[stock_name] = ordered_dates
            if self.generated_return_output_dir is not None:
                self.generated_return_output_dir.mkdir(parents=True, exist_ok=True)
                rows = [
                    {"date": date_key, "log_return": filtered[date_key]}
                    for date_key in ordered_har_dates
                    if date_key in filtered
                ]
                write_csv(
                    self.generated_return_output_dir / (stock_name + "_low_freq_stock_returns.csv"),
                    rows,
                )
            return filtered

        return self._load_return_series(
            self.stock_return_folder,
            stock_name,
            self.stock_return_cache,
        )

    def _load_future_returns(self, future_name):
        if self.future_daily_price_folder is not None and self.future_daily_price_folder.exists():
            path = find_asset_file(
                self.future_daily_price_folder,
                future_name,
                extensions=[".csv"],
                required_substrings=["processed"],
            )
            ordered_dates, close_by_date = load_daily_close_series_from_csv(path)
            return_by_date = {}
            previous_close = None
            for current_date in ordered_dates:
                current_close = close_by_date[current_date]
                if previous_close is not None and previous_close > 0.0:
                    return_by_date[current_date] = float(np.log(current_close) - np.log(previous_close))
                previous_close = current_close
            if not return_by_date:
                raise ValueError(
                    "No usable future daily returns were built for '{0}' from '{1}'.".format(
                        future_name,
                        path,
                    )
                )
            self.future_return_cache[future_name] = return_by_date
            if self.generated_return_output_dir is not None:
                self.generated_return_output_dir.mkdir(parents=True, exist_ok=True)
                rows = [
                    {"date": date_key, "log_return": return_by_date[date_key]}
                    for date_key in ordered_dates
                    if date_key in return_by_date
                ]
                write_csv(
                    self.generated_return_output_dir / (future_name + "_low_freq_future_returns.csv"),
                    rows,
                )
            return return_by_date

        return self._load_return_series(
            self.future_return_folder,
            future_name,
            self.future_return_cache,
        )

    def _get_corr_offset(self, stock_length, corr_length):
        if corr_length > stock_length:
            raise ValueError("Correlation series is longer than stock HAR series.")
        return stock_length - corr_length

    def _get_window_corr_stats(self, stock_name, future_name, start_index, window_length):
        stock_har = self._load_stock_har(stock_name)
        corr_values = self._load_corr_series(future_name, stock_name)
        corr_offset = self._get_corr_offset(len(stock_har["values"]), len(corr_values))
        aligned = []
        for stock_idx in range(start_index, start_index + window_length):
            corr_idx = stock_idx - corr_offset
            if 0 <= corr_idx < len(corr_values):
                corr_value = corr_values[corr_idx]
                if np.isnan(corr_value):
                    continue
                aligned.append(corr_value)
        if not aligned:
            raise ValueError(
                "No overlapping correlation observations for stock '{0}' and future '{1}'."
                .format(stock_name, future_name)
            )
        mean_abs_corr = float(np.mean(np.abs(np.asarray(aligned, dtype=float))))
        return corr_values, corr_offset, mean_abs_corr

    def _select_futures(self, stock_name, start_index, window_length):
        pairing_mode = self.pairing_cfg.get("mode", "auto_max_abs_corr")
        if pairing_mode == "manual":
            manual_map = self.pairing_cfg.get("manual_map", {})
            if stock_name not in manual_map:
                raise ValueError(
                    "pairing.manual_map is missing stock '{0}'.".format(stock_name)
                )
            configured = manual_map[stock_name]
            if isinstance(configured, str):
                future_names = [configured]
            else:
                future_names = list(configured)

            selected = []
            for future_name in future_names:
                corr_values, corr_offset, mean_abs_corr = self._get_window_corr_stats(
                    stock_name,
                    future_name,
                    start_index,
                    window_length,
                )
                selected.append((future_name, corr_values, corr_offset, mean_abs_corr))
            return selected

        if pairing_mode == "all_futures":
            selected = []
            for future_name in self.future_names:
                try:
                    corr_values, corr_offset, mean_abs_corr = self._get_window_corr_stats(
                        stock_name,
                        future_name,
                        start_index,
                        window_length,
                    )
                except Exception:
                    continue
                selected.append((future_name, corr_values, corr_offset, mean_abs_corr))
            if not selected:
                raise ValueError(
                    "Could not build any valid stock-future pairs for stock '{0}'.".format(
                        stock_name,
                    )
                )
            return selected

        best_future = None
        best_corr_values = None
        best_corr_offset = None
        best_score = -1.0
        for future_name in self.future_names:
            try:
                corr_values, corr_offset, mean_abs_corr = self._get_window_corr_stats(
                    stock_name,
                    future_name,
                    start_index,
                    window_length,
                )
            except Exception:
                continue
            if mean_abs_corr > best_score:
                best_future = future_name
                best_corr_values = corr_values
                best_corr_offset = corr_offset
                best_score = mean_abs_corr

        if best_future is None:
            raise ValueError(
                "Could not auto-select a future for stock '{0}'.".format(stock_name)
            )
        return [(best_future, best_corr_values, best_corr_offset, best_score)]

    def _compute_hedging_detail(
        self,
        model_name,
        stock_name,
        future_name,
        stock_har,
        future_har,
        corr_values,
        corr_offset,
        start_index,
        predicted_values,
        stock_returns,
        future_returns,
    ):
        lag_steps = int(self.hedging_cfg.get("lag_steps", 1))
        beta_min = self.hedging_cfg.get("beta_clip_min")
        beta_max = self.hedging_cfg.get("beta_clip_max")

        stock_return_values = []
        hedged_return_values = []
        beta_values = []
        used_corr_values = []
        used_dates = []
        future_sigma_series = self._build_rolling_sigma_series_from_returns(
            stock_har["dates"],
            future_returns,
            self.sigma_window,
        )

        for local_idx, predicted_value in enumerate(predicted_values):
            stock_idx = start_index + local_idx
            previous_idx = stock_idx - lag_steps
            if previous_idx < 0:
                continue

            current_date = stock_har["dates"][stock_idx]
            stock_return = stock_returns.get(current_date)
            future_return = future_returns.get(current_date)
            corr_idx = previous_idx - corr_offset
            future_sigma_value = future_sigma_series[previous_idx]

            if stock_return is None or future_return is None:
                continue
            if corr_idx < 0 or corr_idx >= len(corr_values):
                continue
            corr_value = float(corr_values[corr_idx])
            if np.isnan(corr_value) or not np.isfinite(future_sigma_value):
                continue

            predicted_sigma = to_sigma(
                predicted_value,
                self.prediction_value_type,
                self.variance_epsilon,
            )
            if future_sigma_value <= 1e-8:
                beta_value = 0.0
            else:
                beta_value = corr_value * predicted_sigma / future_sigma_value
            if beta_min is not None:
                beta_value = max(beta_value, float(beta_min))
            if beta_max is not None:
                beta_value = min(beta_value, float(beta_max))

            stock_return_values.append(float(stock_return))
            hedged_return_values.append(float(stock_return) - beta_value * float(future_return))
            beta_values.append(beta_value)
            used_corr_values.append(corr_value)
            used_dates.append(current_date)

        if len(stock_return_values) < 2:
            stock_return_variance = float("nan")
            hedged_return_variance = float("nan")
            hedging_effectiveness = float("nan")
        else:
            stock_return_variance = float(np.var(stock_return_values, ddof=1))
            hedged_return_variance = float(np.var(hedged_return_values, ddof=1))
            hedging_effectiveness = 1.0 - (hedged_return_variance / stock_return_variance)

        return {
            "model": model_name,
            "stock": stock_name,
            "future": future_name,
            "num_observations": len(stock_return_values),
            "window_start_date": used_dates[0] if used_dates else "",
            "window_end_date": used_dates[-1] if used_dates else "",
            "mean_beta": nanmean(beta_values),
            "mean_abs_beta": nanmean(np.abs(np.asarray(beta_values, dtype=float))) if beta_values else float("nan"),
            "mean_corr": nanmean(used_corr_values),
            "mean_abs_corr": nanmean(np.abs(np.asarray(used_corr_values, dtype=float))) if used_corr_values else float("nan"),
            "stock_return_variance": stock_return_variance,
            "hedged_return_variance": hedged_return_variance,
            "hedging_effectiveness": hedging_effectiveness,
        }

    def validate_inputs(self):
        print("--- Checking low-frequency hedging inputs")
        required_folders = [
            self.stock_har_folder,
            self.energy_har_folder,
        ]
        if self.corr_mode != "rolling_return":
            required_folders.append(self.corr_folder)
        for folder in required_folders:
            if not folder.exists():
                raise FileNotFoundError("Required folder not found: '{0}'.".format(folder))

        if self.stock_pivoted_price_folder is None:
            if not self.stock_return_folder.exists():
                raise FileNotFoundError("Required folder not found: '{0}'.".format(self.stock_return_folder))
        elif not self.stock_pivoted_price_folder.exists():
            raise FileNotFoundError(
                "Stock pivoted price folder not found: '{0}'.".format(
                    self.stock_pivoted_price_folder,
                )
            )

        if self.future_daily_price_folder is None:
            if not self.future_return_folder.exists():
                raise FileNotFoundError("Required folder not found: '{0}'.".format(self.future_return_folder))
        elif not self.future_daily_price_folder.exists():
            raise FileNotFoundError(
                "Future daily price folder not found: '{0}'.".format(
                    self.future_daily_price_folder,
                )
            )
        if self.future_pivoted_price_folder is not None and not self.future_pivoted_price_folder.exists():
            raise FileNotFoundError(
                "Future pivoted price folder not found: '{0}'.".format(
                    self.future_pivoted_price_folder,
                )
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

        hedging_rows = []
        pairing_rows = []
        alignment_rows = []

        for model_name, model_cfg in self.models_cfg.items():
            workbook_path = str(resolve_path(model_cfg["prediction_workbook"], self.project_root))
            predictions = self._load_prediction_workbook(workbook_path)
            print("--- Running low-frequency hedging for model: {0}".format(model_name))

            for stock_name, prediction_data in predictions.items():
                stock_har = self._load_stock_har(stock_name)
                actual_array = np.asarray(prediction_data["actual"], dtype=float)
                predicted_values = list(prediction_data["predicted"])
                start_index = find_matching_window(
                    stock_har["values"],
                    actual_array,
                    self.match_atol,
                    self.match_rtol,
                )
                end_index = start_index + len(predicted_values) - 1

                stock_returns = self._load_stock_returns(stock_name)
                selected_futures = self._select_futures(
                    stock_name,
                    start_index,
                    len(predicted_values),
                )

                for future_name, corr_values, corr_offset, mean_abs_corr in selected_futures:
                    future_har = self._load_future_har(future_name)
                    future_returns = self._load_future_returns(future_name)

                    alignment_rows.append(
                        {
                            "model": model_name,
                            "stock": stock_name,
                            "future": future_name,
                            "prediction_workbook": workbook_path,
                            "har_start_index": start_index,
                            "har_end_index": end_index,
                            "har_start_date": stock_har["dates"][start_index],
                            "har_end_date": stock_har["dates"][end_index],
                            "prediction_rows": len(predicted_values),
                            "dataset_labels": "|".join(sorted(set(prediction_data["dataset"]))),
                            "mean_abs_corr_selected": mean_abs_corr,
                        }
                    )
                    pairing_rows.append(
                        {
                            "model": model_name,
                            "stock": stock_name,
                            "future": future_name,
                            "pairing_mode": self.pairing_cfg.get("mode", "auto_max_abs_corr"),
                            "mean_abs_corr": mean_abs_corr,
                        }
                    )
                    hedging_rows.append(
                        self._compute_hedging_detail(
                            model_name,
                            stock_name,
                            future_name,
                            stock_har,
                            future_har,
                            corr_values,
                            corr_offset,
                            start_index,
                            predicted_values,
                            stock_returns,
                            future_returns,
                        )
                    )

        hedging_summary = summarize_groups(
            hedging_rows,
            group_keys=["model"],
            numeric_keys=[
                "num_observations",
                "mean_beta",
                "mean_abs_beta",
                "mean_corr",
                "mean_abs_corr",
                "hedging_effectiveness",
            ],
        )

        write_csv(self.output_dir / "hedging_detail.csv", hedging_rows)
        write_csv(self.output_dir / "hedging_summary.csv", hedging_summary)
        write_csv(self.output_dir / "selected_pairs.csv", pairing_rows)
        if self.output_cfg.get("write_alignment_detail", True):
            write_csv(self.output_dir / "alignment_detail.csv", alignment_rows)

        print("--- Finished low-frequency hedging")


def load_config(config_path):
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    required_sections = [
        "data",
        "evaluation",
        "pairing",
        "hedging",
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
        description="Run low-frequency hedging effectiveness tests."
    )
    parser.add_argument(
        "--config",
        default="config/low_freq_hedging.yaml",
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
    runner = LowFreqHedgingRunner(config, config_path)
    if args.check_only:
        runner.validate_inputs()
        print("--- Check completed")
        return
    runner.run()


if __name__ == "__main__":
    main()
