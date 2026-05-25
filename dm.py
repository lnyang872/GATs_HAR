import glob
import os
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


def dm_test_from_diffs(d):
    d = np.asarray(d)
    t = len(d)
    d_mean = np.mean(d)

    sample_variance_d = np.var(d, ddof=1)
    var_d_mean = sample_variance_d / t
    dm_statistic = d_mean / np.sqrt(var_d_mean)
    p_value = 2 * stats.norm.sf(np.abs(dm_statistic))
    return dm_statistic, p_value


def perform_overall_dm_analysis_to_excel():
    """
    Load all model outputs, run an aggregate DM test, and export the result to Excel.
    """
    file_path = r"D:\GATHAR\results_high_with_energy"
    all_files = glob.glob(os.path.join(file_path, "*.xlsx"))

    if not all_files:
        print("Error: no .xlsx files were found in the current directory.")
        return

    print(f"Found {len(all_files)} model files: {all_files}\n")

    # Step 1: load and preprocess all data.
    all_data = {}
    model_names_from_files = []
    for file in all_files:
        model_name = os.path.splitext(os.path.basename(file))[0]
        model_names_from_files.append(model_name)
        try:
            xls = pd.ExcelFile(file)
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name)
                df.columns = [col.strip() for col in df.columns]

                # Keep these original Chinese column names because they must match
                # the exported workbook format.
                required_cols = ["真实值", "预测值", "数据集"]
                if not all(col in df.columns for col in required_cols):
                    continue

                df_processed = df[required_cols]
                df_reversed = df_processed.iloc[::-1].reset_index(drop=True)

                if sheet_name not in all_data:
                    all_data[sheet_name] = {}
                all_data[sheet_name][model_name] = df_reversed
        except Exception as e:
            print(f"Error while processing file {file}: {e}")

    stock_names = list(all_data.keys())
    if not stock_names:
        print("No data were loaded successfully, so the analysis cannot continue.")
        return

    # Sort model names so the output table has a stable order.
    model_names = sorted(model_names_from_files)

    # Create an Excel writer.
    output_filename = "DM_Test_Results_h_0516.xlsx"
    with pd.ExcelWriter(output_filename, engine="openpyxl") as writer:
        # Step 2: run the aggregate test for Validation and Test separately.
        for dataset_type in ["Validation", "Test"]:
            print(f"{'=' * 25} Generating DM test table for the {dataset_type} set {'=' * 25}")

            # Create an empty DataFrame for results.
            dm_table = pd.DataFrame(index=model_names, columns=model_names, dtype=object)

            model_pairs = combinations(model_names, 2)

            for model1, model2 in model_pairs:
                combined_loss_diffs = []

                # Step 3: collect loss-difference series across all stocks.
                for stock in stock_names:
                    if model1 not in all_data[stock] or model2 not in all_data[stock]:
                        continue

                    df1 = all_data[stock][model1]
                    df2 = all_data[stock][model2]

                    # Find common indices across all available models for this stock.
                    indices_sets = [
                        set(all_data[stock][m][all_data[stock][m]["数据集"] == dataset_type].index)
                        for m in model_names
                        if m in all_data[stock]
                    ]
                    if not indices_sets:
                        continue
                    common_indices = sorted(list(set.intersection(*indices_sets)))

                    if len(common_indices) < 2:
                        continue

                    actuals = df1.loc[common_indices, "真实值"].values
                    pred1 = df1.loc[common_indices, "预测值"].values
                    pred2 = df2.loc[common_indices, "预测值"].values

                    # Loss difference: d = e1^2 - e2^2.
                    loss_diff = np.abs(actuals - pred1) ** 2 - np.abs(actuals - pred2) ** 2
                    combined_loss_diffs.append(loss_diff)

                # Step 4: concatenate and run one DM test.
                if not combined_loss_diffs:
                    continue

                final_d_series = np.concatenate(combined_loss_diffs)
                dm_stat, p_value = dm_test_from_diffs(final_d_series)

                # Step 5: format the result and fill the table.
                if not np.isnan(dm_stat):
                    significance_star = "*" if p_value < 0.05 else ""
                    formatted_stat = f"{dm_stat:.2f}{significance_star}"
                    dm_table.loc[model1, model2] = formatted_stat
                else:
                    dm_table.loc[model1, model2] = "N/A"

            # Step 6: print and save to Excel.
            print(f"\n--- {dataset_type} Set Results ---")
            print(dm_table.fillna("").to_string())

            dm_table.to_excel(writer, sheet_name=f"{dataset_type}_DM_Test", index=True)
            print(
                f"\nResults have been written to the '{dataset_type}_DM_Test' sheet "
                f"of '{output_filename}'."
            )

    print(f"\n{'=' * 20} All analyses completed {'=' * 20}")


if __name__ == "__main__":
    perform_overall_dm_analysis_to_excel()
