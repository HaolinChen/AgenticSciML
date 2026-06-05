"""
Phase 0.5: Data Analysis

Thin wrapper script for running exploratory data analysis on training datasets.
This phase runs BEFORE contract creation if training data is provided.
"""

import os
import sys
import json
import argparse

from constants import DATASET_CONFIG_PATH, USER_INPUT_DIR
from data_analyst import run_analysis_workflow


def main():
    """
    Main entry point for data analysis phase.

    Returns:
        0 on success, non-zero on failure
    """
    print("\n" + "="*80)
    print("PHASE 0.5: DATA ANALYSIS")
    print("="*80)

    # Check if dataset config exists
    if not os.path.exists(DATASET_CONFIG_PATH):
        print("No dataset_config.json found in USER_INPUT/")
        print("Skipping data analysis phase")
        return 0

    # Load config
    try:
        with open(DATASET_CONFIG_PATH, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in dataset_config.json: {e}")
        return 1

    # Check if training set exists
    if "training_set" not in config:
        print("No training_set found in dataset_config.json")
        print("Skipping data analysis phase")
        return 0

    # Verify training file exists
    training_file = config["training_set"].get("filename", "")
    training_path = os.path.join(USER_INPUT_DIR, training_file)

    if not os.path.exists(training_path):
        print(f"Error: Training file not found: {training_path}")
        print(f"Specified in dataset_config.json: {training_file}")
        return 1

    print(f"\nTraining dataset detected: {training_file}")
    print("Starting exploratory data analysis...\n")

    # Run analysis workflow
    result = run_analysis_workflow(max_debug_iterations=3)

    if result == 0:
        print("\n✓ Phase 0.5 complete: Data analysis report generated")
    else:
        print("\n⚠ Phase 0.5 warning: Data analysis failed")
        print("Continuing without analysis report...")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run exploratory data analysis on training dataset"
    )
    args = parser.parse_args()

    sys.exit(main())
