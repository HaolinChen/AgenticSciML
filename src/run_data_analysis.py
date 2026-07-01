"""
Phase 0.5: Data Analysis

Thin wrapper script for running exploratory data analysis on training datasets.
This phase runs BEFORE contract creation if training data is provided.

============================================================================
中文模块说明（学习注释）
============================================================================
作用：
    Phase 0.5“数据分析”阶段的轻量入口脚本（thin wrapper）。它负责做前置校验
    （配置是否存在、是否配置了训练集、训练文件是否真实存在），随后把真正的
    探索性数据分析工作委托给 data_analyst.run_analysis_workflow。

在 pipeline 中的位置：
    位于合约生成（Phase 1）之前。既可被 main.py 的 run_data_analysis_phase()
    以函数方式调用（from run_data_analysis import main），也可作为独立脚本运行。

主要输入：
    - 文件：DATASET_CONFIG_PATH（USER_INPUT/dataset_config.json）及其中
      training_set.filename 指向的训练集文件（位于 USER_INPUT_DIR）。
    - 命令行：作为脚本运行时支持 argparse（当前无额外参数）。

主要输出：
    - 产物：由 run_analysis_workflow 生成的数据分析报告（写入 DATA_ANALYSIS 目录）。
    - 返回值/退出码：main() 返回 0 表示成功或“合理跳过”，非 0 表示失败；
      作为脚本运行时以该返回值作为进程退出码（sys.exit）。

关键函数列表：
    - main()  执行前置校验并调用 run_analysis_workflow，返回状态码。
============================================================================
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

    中文：数据分析阶段主入口。
    做什么：依次校验 dataset_config.json 是否存在、是否含 training_set、训练文件是否存在，
        全部通过后调用 run_analysis_workflow(max_debug_iterations=3) 执行分析工作流。
    关键返回语义（重要）：
        - 缺少配置或缺少 training_set：视为“合理跳过”，返回 0（不报错）。
        - JSON 非法 / 训练文件缺失：返回 1（失败）。
        - 否则返回 run_analysis_workflow 的返回码。
    副作用：向标准输出打印阶段进度与结果提示。
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
