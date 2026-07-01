"""
Generate comprehensive performance analysis for a solution.

This script can be called from command line or imported by other scripts.
It runs the analyst agent on a completed solution.

================================================================================
中文模块说明（学习注释）
================================================================================
作用：
    本模块是 AgenticSciML **Phase3 进化循环** 的“分析(Analyst)”环节，也用于 Phase2 根解。
    读取某个已完成解(solution)的源码、训练/评测日志与产物图片，调用 Analyst 智能体
    生成一份综合性能分析报告，落盘到分析库(AB)。该报告后续会被检索记忆(retrieve_AB/KB)
    复用，为选择与提案提供依据。

所属阶段：
    Phase3（进化循环）：选亲本 → 多智能体辩论提案 → 工程化子代 → **分析(本模块)**。
    （Phase2 生成根解后同样会调用它做首份分析。）

主要输入：
    - 参数：solution_id（要分析的解 ID）、proposal_text（该解对应的提案文本，根解为 None）、
        parent_id（亲本 ID，用于对比，根解为 None）。
    - 文件（均在 {SOLUTION_AND_OUTPUTS_DIR}/{solution_id}/ 下）：
        * solution.py     —— 解源码（必需）
        * train_log.txt   —— 训练日志（必需）
        * test_log.txt    —— 评测日志（必需，末尾含 JSON 结果，用于抽取 score）
        * *.png/*.jpg/*.jpeg —— 产物图片（可选，作为多模态输入给 Analyst）
        * {AB_DIR}/{parent_id}_analysis.md —— 亲本分析报告（可选，用于对比）
    - LLM：analyst_agent（支持传入图片做多模态分析）。

主要输出：
    - 文件：{AB_DIR}/{solution_id}_analysis.md —— 分析报告 markdown。
    - 返回：analyze_solution 返回分析报告文本。

关键函数清单：
    - analyze_solution：装配输入(代码/日志/图片/分数/亲本分析)→调用 Analyst→保存报告。
    - main：命令行入口。
================================================================================
"""

import argparse
import os
import json
import re
import time
import base64
import glob
from agents import analyst_agent
from constants import SOLUTION_AND_OUTPUTS_DIR, AB_DIR


def analyze_solution(solution_id: str, proposal_text: str | None = None,
                     parent_id: str | None = None):
    """
    Analyze a solution and generate comprehensive report.

    Args:
        solution_id: Solution ID (e.g., "solution_0", "solution_00")
        proposal_text: Proposal markdown content (None for root solution)
        parent_id: Parent solution ID (None for root solution)

    Returns:
        Analysis markdown text

    中文：本模块主入口。读取解源码、训练/评测日志、产物图片，从评测日志抽取 score，
    可选读取亲本分析用于对比，然后调用 analyst_agent 生成分析报告并落盘到 AB 库。
    副作用：读取解目录下多种文件；调用 LLM(analyst_agent)；写 {AB_DIR}/{solution_id}_analysis.md。
    异常：缺少 solution.py / train_log.txt / test_log.txt 会抛 FileNotFoundError。
    """
    print(f"\n{'='*80}")
    print(f"ANALYZING: {solution_id}")
    print(f"{'='*80}\n")

    # Read solution code
    solution_dir = f"{SOLUTION_AND_OUTPUTS_DIR}/{solution_id}"
    solution_path = f"{solution_dir}/solution.py"

    if not os.path.exists(solution_path):
        raise FileNotFoundError(f"Solution not found: {solution_path}")

    with open(solution_path, 'r') as f:
        solution_code = f.read()

    print(f"Loaded solution code ({len(solution_code)} chars)")

    # Read logs
    train_log_path = f"{solution_dir}/train_log.txt"
    test_log_path = f"{solution_dir}/test_log.txt"

    if not os.path.exists(train_log_path):
        raise FileNotFoundError(f"Training log not found: {train_log_path}")
    if not os.path.exists(test_log_path):
        raise FileNotFoundError(f"Testing log not found: {test_log_path}")

    with open(train_log_path, 'r') as f:
        train_log = f.read()

    with open(test_log_path, 'r') as f:
        test_log = f.read()

    print(f"Loaded train log ({len(train_log)} chars)")
    print(f"Loaded test log ({len(test_log)} chars)")

    # Check for plots in solution directory
    # 中文：扫描解目录下的图片产物（png/jpg/jpeg），读取为 bytes 并做 base64 编码，
    # 以 (文件名, base64) 列表传给 Analyst 做多模态分析；单张读取失败仅告警、不中断。
    plot_images = []
    plot_pattern_extensions = ['*.png', '*.jpg', '*.jpeg']
    for ext in plot_pattern_extensions:
        plot_files = glob.glob(os.path.join(solution_dir, ext))
        for plot_path in plot_files:
            try:
                with open(plot_path, 'rb') as f:
                    image_bytes = f.read()
                    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
                    plot_filename = os.path.basename(plot_path)
                    plot_images.append((plot_filename, image_b64))
            except Exception as e:
                print(f"Warning: Could not load plot {plot_path}: {e}")

    if plot_images:
        print(f"Found {len(plot_images)} plot(s) for analysis")
    else:
        print("No plots found in solution directory")

    # Extract score from test_log (parse JSON at end)
    # 中文：评测日志末尾会打印形如 {"status": ..., "score": ...} 的 JSON，
    # 用正则抓取并解析出 score；抓不到则退化为 inf（表示无有效分数）。
    json_match = re.search(r'\{"status".*?\}', test_log)
    if json_match:
        result_json = json.loads(json_match.group())
        score = result_json.get("score", float('inf'))
        print(f"Extracted score: {score}")
    else:
        print("Warning: Could not extract score from test log, using inf")
        score = float('inf')

    # Read parent analysis if exists
    # 中文：若提供 parent_id，则读取亲本分析报告（检索记忆的一种），供 Analyst 做代际对比。
    parent_analysis = None
    if parent_id:
        parent_analysis_path = f"{AB_DIR}/{parent_id}_analysis.md"
        if os.path.exists(parent_analysis_path):
            with open(parent_analysis_path, 'r') as f:
                parent_analysis = f.read()
            print(f"Loaded parent analysis ({len(parent_analysis)} chars)")
        else:
            print(f"Warning: Parent analysis not found: {parent_analysis_path}")

    # Call analyst agent
    # 中文：调用 Analyst 智能体(LLM)，综合代码、训练/评测日志、分数、提案、亲本分析与图片，
    # 产出结构化分析报告文本。
    print("\nCalling analyst agent (this may take a moment)...")
    analysis_markdown = analyst_agent(
        solution_code=solution_code,
        train_log=train_log,
        test_log=test_log,
        score=score,
        proposal=proposal_text,
        parent_analysis=parent_analysis,
        plot_images=plot_images if plot_images else None
    )

    print(f"Analysis generated ({len(analysis_markdown)} chars)")

    # Save analysis
    # 中文：报告落盘到 AB(分析库)，命名为 {solution_id}_analysis.md，成为后续检索记忆的一部分。
    os.makedirs(AB_DIR, exist_ok=True)
    analysis_path = f"{AB_DIR}/{solution_id}_analysis.md"

    with open(analysis_path, 'w') as f:
        f.write(analysis_markdown)

    print(f"\n✓ Analysis saved to: {analysis_path}")

    return analysis_markdown


def main():
    """Command-line interface for analyze.py

    中文：命令行入口。解析 --solution_id / --parent_id / --proposal_file，
    可选读取提案文件后调用 analyze_solution 生成分析，并在末尾打印报告预览。
    """
    parser = argparse.ArgumentParser(description="Analyze solution performance")
    parser.add_argument("--solution_id", type=str, required=True,
                        help="Solution ID (e.g., solution_0)")
    parser.add_argument("--parent_id", type=str, default=None,
                        help="Parent solution ID for comparison")
    parser.add_argument("--proposal_file", type=str, default=None,
                        help="Path to proposal markdown file")
    args = parser.parse_args()

    # Read proposal if provided
    proposal_text = None
    if args.proposal_file:
        if os.path.exists(args.proposal_file):
            with open(args.proposal_file, 'r') as f:
                proposal_text = f.read()
            print(f"Loaded proposal ({len(proposal_text)} chars)")
        else:
            print(f"Warning: Proposal file not found: {args.proposal_file}")

    # Generate analysis
    analysis = analyze_solution(
        solution_id=args.solution_id,
        proposal_text=proposal_text,
        parent_id=args.parent_id
    )

    # Print preview
    print("\n" + "="*80)
    print("ANALYSIS PREVIEW")
    print("="*80)
    print(analysis)
    print("="*80)


if __name__ == "__main__":
    main()
