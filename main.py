"""
Single unified entry point for the DAIC-WOZ Multimodal Depression Detection project.

Supported Commands:
  python main.py build        -> runs build_dataset.main()
  python main.py train        -> runs train_evaluate.main()
  python main.py fusion       -> runs fusion.main()
  python main.py regression   -> runs regression.main()
  python main.py fairness     -> runs fairness.main()
  python main.py all          -> runs build, train, fusion, regression, fairness in order
  python main.py mock         -> runs tests/generate_mock_data.main() to create fake smoke-test data
"""

import sys
from pathlib import Path

# Add project root to sys.path to guarantee imports work cleanly
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src import build_dataset, train_evaluate, fusion, regression, fairness
from src.config import MOCK_RAW_DATA_DIR
from tests import generate_mock_data


def print_usage():
    """Print available CLI commands."""
    print("""
========================================================================
DAIC-WOZ Multimodal Depression Detection Pipeline
========================================================================
Usage: python main.py <command>

Available commands:
  mock        Generate synthetic mock DAIC-WOZ cohort for smoke testing
  build       Extract multimodal features and compile data/processed/feature_table.csv
  train       Train and evaluate baseline classification models on dev split
  fusion      Compare unimodal (text/audio/visual) vs early vs late fusion
  regression  Train continuous PHQ-8 severity score regressors
  fairness    Audit trained classifier for demographic disparity across gender
  all         Run full pipeline: build -> train -> fusion -> regression -> fairness
========================================================================
""")


def run_all():
    """Execute complete end-to-end pipeline in sequential order."""
    mock_split = MOCK_RAW_DATA_DIR / "train_split_Depression_AVEC2017.csv"
    data_root = MOCK_RAW_DATA_DIR if mock_split.exists() else None
    if data_root is not None:
        print(f"[INFO] Using isolated smoke-test cohort in {data_root}")
    print("\n>>> STEP 1/5: Compiling Processed Multimodal Feature Table...")
    build_dataset.main(data_root)

    print("\n>>> STEP 2/5: Training and Evaluating Multimodal Classifiers...")
    train_evaluate.main()

    print("\n>>> STEP 3/5: Benchmarking Multimodal Fusion Strategies...")
    fusion.main()

    print("\n>>> STEP 4/5: Training Continuous PHQ-8 Severity Regressors...")
    regression.main()

    print("\n>>> STEP 5/5: Auditing Demographic Fairness Across Gender...")
    fairness.main()

    print("\n" + "=" * 70)
    print("ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!")
    print("Results and visualizations are stored in results/")
    print("Launch the web dashboard via: python app.py")
    print("=" * 70)


def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    command = sys.argv[1].strip().lower()

    command_map = {
        "mock": generate_mock_data.main,
        "build": build_dataset.main,
        "train": train_evaluate.main,
        "fusion": fusion.main,
        "regression": regression.main,
        "fairness": fairness.main,
        "all": run_all,
    }

    action = command_map.get(command)
    if action is None:
        print(f"[ERROR] Unknown command '{command}'.")
        print_usage()
        sys.exit(1)

    action()


if __name__ == "__main__":
    main()
