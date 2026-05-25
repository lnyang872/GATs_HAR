import os
from datetime import datetime

import pandas as pd

from run_ablation_suite_single_task import load_base_config, run_objective_suite


BASE_CONFIG_PATH = "config/ablation_full.yaml"
OUTPUT_ROOT = "output/ablation_high_only"


def main():
    base_config = load_base_config(BASE_CONFIG_PATH)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suite_root = os.path.join(OUTPUT_ROOT, f"suite_{timestamp}")
    os.makedirs(suite_root, exist_ok=True)

    results = run_objective_suite(
        base_config=base_config,
        objective="high_only",
        suite_root=suite_root,
    )

    results_df = pd.DataFrame(results)
    results_path = os.path.join(suite_root, "ablation_summary.csv")
    results_df.to_csv(results_path, index=False, float_format="%.12f")
    print(f"\nHigh-only ablation summary saved to: {results_path}")


if __name__ == "__main__":
    main()
