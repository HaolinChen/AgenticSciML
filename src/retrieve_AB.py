"""
Retrieve analysis reports (parent, siblings, and uncles) for the champion solution.

This is a deterministic script (no LLM) that finds analysis reports
of the champion's parent, siblings, and uncles (parent's siblings).

============================================================================
中文模块说明（学习注释）
============================================================================
作用：
    检索“分析记忆库/尝试记忆库”（Analysis Bank，目录 AB/）中与当前 champion 相关的
    历史分析报告，具体包括其“父解、兄弟解、叔伯解（父解的兄弟）”三类近亲的分析。
    这些过往经验作为上下文，帮助智能体理解“同一支系里试过什么、成败原因”，避免重复弯路。

在 RAG / pipeline 中的角色：
    RAG 检索增强记忆组件之一。是一个纯确定性脚本（不调用 LLM），仅靠解 id 的命名结构与
    results.json 中的 parent_id 关系来推断亲缘，然后读取对应的分析 markdown。

亲缘关系约定（关键）：
    - 解 id 形如 "solution_" + 数字串，数字串编码了进化树路径。
    - 父解：去掉数字串的最后一位（如 solution_012 → solution_01；根解 solution_0 无父）。
    - 兄弟解：与自己拥有相同 parent_id 的其它解（依据 results.json 中的 parent_id 字段）。
    - 叔伯解：父解的兄弟，即与“父解的 parent_id（祖父）”相同的其它解（排除父解自身）。

主要输入：
    - RESULTS_FILE（默认 ./RESULTS/results.json）：所有解的元数据，含 parent_id，用于查兄弟/叔伯。
    - AB_DIR/{solution_id}_analysis.md：各解的分析报告文件。
    - 参数 champion_id：要检索的中心解 id。

主要输出：
    retrieve_AB(champion_id) 返回字典：parent_id、parent_analysis、
    sibling_analyses（[{sibling_id, analysis}]）、uncle_analyses（[{uncle_id, analysis}]）、summary。

关键函数列表：
    - get_parent_id(solution_id)        由 id 命名结构推算父解 id（去掉末位数字）
    - find_siblings(champion_id,results) 找同父的兄弟解 id 列表（排除自身）
    - find_uncles(champion_id,results)   找父解的兄弟（叔伯）id 列表（排除父解）
    - read_analysis(solution_id)         读取某解的分析 md，缺失返回 None
    - retrieve_AB(champion_id)           汇总父/兄弟/叔伯分析并生成摘要
    - main()                             命令行自测入口

副作用：仅读取上述文件；不调用 LLM。
============================================================================
"""

import json
import os
import argparse
from constants import RESULTS_FILE, AB_DIR


def get_parent_id(solution_id: str) -> str | None:
    """
    Compute parent ID for a given solution by removing the last digit.

    Parent: Remove last digit (e.g., "solution_012" → "solution_01")

    Args:
        solution_id: Solution ID (e.g., "solution_0", "solution_01", "solution_012")

    Returns:
        Parent solution ID, or None if this is the root solution

    中文：仅凭 id 命名结构推算父解 id（不查 results.json）。规则：去掉数字串的最后一位。
        根解 solution_0 无父，返回 None。id 格式非法（不以 "solution_" 开头）时抛 ValueError。
    """
    # 中文：取出 "solution_" 之后的数字串（如 "solution_012" → "012"）
    if not solution_id.startswith("solution_"):
        raise ValueError(f"Invalid solution ID format: {solution_id}")

    numeric_id = solution_id.replace("solution_", "")

    # 中文：根解（数字串为 "0"）没有父解
    if numeric_id == "0":
        return None

    # 中文：去掉数字串末位即得父解的数字串，再拼回前缀
    parent_numeric_id = numeric_id[:-1]
    return f"solution_{parent_numeric_id}"


def find_siblings(champion_id: str, results: dict) -> list:
    """
    Find all siblings of the champion solution.

    Siblings are solutions that share the same parent.

    Args:
        champion_id: Solution ID
        results: Dictionary of all solution results

    Returns:
        List of sibling IDs (excluding champion itself)

    中文：查“兄弟解”——与 champion 拥有相同 parent_id 的其它解。这里的 parent_id 直接取自
        results.json（而非命名推算）。champion 不在 results 中或为根解（无父）时返回空列表。
    """
    # 中文：先从 results 里拿到 champion 自身的记录与其 parent_id
    champion_data = results.get(champion_id)
    if not champion_data:
        return []

    parent_id = champion_data.get("parent_id")

    # 中文：根解没有父解，也就没有兄弟
    if parent_id is None:
        return []

    # 中文：遍历所有解，筛出 parent_id 相同且不是自己的，即为兄弟
    siblings = [
        sol_id for sol_id, sol_data in results.items()
        if sol_data.get("parent_id") == parent_id and sol_id != champion_id
    ]

    return siblings


def find_uncles(champion_id: str, results: dict) -> list:
    """
    Find all uncles (parent's siblings) of the champion solution.

    Uncles are solutions that share the same parent as the champion's parent.

    Args:
        champion_id: Solution ID
        results: Dictionary of all solution results

    Returns:
        List of uncle IDs (parent's siblings, excluding parent itself)

    中文：查“叔伯解”——即父解的兄弟。做法：先找到父解，再找祖父，最后筛出所有 parent_id
        等于祖父且不是父解本身的解。以下任一情况返回空列表：champion 不在 results、champion 为
        根解（无父）、或父解为一代解（祖父为根，父解本身即根的孩子而无叔伯）。
    """
    # 中文：取 champion 记录及其 parent_id（父解）
    champion_data = results.get(champion_id)
    if not champion_data:
        return []

    parent_id = champion_data.get("parent_id")

    # 中文：根解无父，自然也无叔伯
    if parent_id is None:
        return []

    # 中文：再取父解的记录，以便找到祖父（父解的 parent_id）
    parent_data = results.get(parent_id)
    if not parent_data:
        return []

    grandparent_id = parent_data.get("parent_id")

    # 中文：若祖父为 None，说明父解是根解的直接孩子（第一代），没有叔伯
    if grandparent_id is None:
        return []

    # 中文：筛出与父解同祖父、且不是父解自身的解，即父解的兄弟（叔伯）
    uncles = [
        sol_id for sol_id, sol_data in results.items()
        if sol_data.get("parent_id") == grandparent_id and sol_id != parent_id
    ]

    return uncles


def read_analysis(solution_id: str) -> str | None:
    """
    Read analysis markdown file for a solution.

    Args:
        solution_id: Solution ID

    Returns:
        Analysis content as string, or None if file doesn't exist

    中文：读取某解在 AB/ 目录下的分析报告（命名约定 {solution_id}_analysis.md）。
        文件不存在时打印警告并返回 None（非致命，调用方会据此跳过）。副作用：读文件、打印日志。
    """
    analysis_path = f"{AB_DIR}/{solution_id}_analysis.md"

    if not os.path.exists(analysis_path):
        print(f"Warning: Analysis file not found: {analysis_path}")
        return None

    with open(analysis_path, 'r') as f:
        return f.read()


def retrieve_AB(champion_id: str):
    """
    Retrieve analysis reports (parent, siblings, and uncles).

    Args:
        champion_id: Champion solution ID

    Returns:
        Dictionary with:
        - parent_id: Parent solution ID (or None)
        - parent_analysis: Analysis of parent solution (or None)
        - sibling_analyses: List of dictionaries with sibling_id and analysis content
        - uncle_analyses: List of dictionaries with uncle_id and analysis content
        - summary: Human-readable summary

    中文：检索总入口。综合命名推算与 results.json 关系，收集 champion 的父、兄弟、叔伯的
        分析报告并生成可读摘要。
    参数：champion_id — 中心解 id。
    返回：字典（见上），其中 parent_analysis 可能为 None；兄弟/叔伯列表仅包含成功读到分析的项。
    副作用：读取 RESULTS_FILE 与 AB_DIR 下多个 md；不调用 LLM。RESULTS_FILE 缺失抛 FileNotFoundError。
    """
    # 中文：读取 results.json —— 用于依据 parent_id 关系查找兄弟与叔伯
    if not os.path.exists(RESULTS_FILE):
        raise FileNotFoundError(f"Results file not found: {RESULTS_FILE}")

    with open(RESULTS_FILE, 'r') as f:
        results = json.load(f)

    # 中文：父解 id 由命名结构推算（去末位数字）
    parent_id = get_parent_id(champion_id)

    # 中文：兄弟与叔伯则依赖 results.json 里的 parent_id 关系来查找
    sibling_ids = find_siblings(champion_id, results)
    uncle_ids = find_uncles(champion_id, results)

    # 中文：读取父解分析（若存在父解）
    parent_analysis = None
    if parent_id:
        parent_analysis = read_analysis(parent_id)

    # 中文：逐个读取兄弟解分析，只保留成功读到内容的
    sibling_analyses = []
    for sibling_id in sibling_ids:
        analysis_content = read_analysis(sibling_id)
        if analysis_content:
            sibling_analyses.append({
                "sibling_id": sibling_id,
                "analysis": analysis_content
            })

    # 中文：逐个读取叔伯解分析，只保留成功读到内容的
    uncle_analyses = []
    for uncle_id in uncle_ids:
        analysis_content = read_analysis(uncle_id)
        if analysis_content:
            uncle_analyses.append({
                "uncle_id": uncle_id,
                "analysis": analysis_content
            })

    # 中文：拼装人类可读的摘要，逐类列出父/兄弟/叔伯的检索情况与各自分析字符数
    summary_lines = [f"Champion: {champion_id}"]

    if parent_id:
        if parent_analysis:
            summary_lines.append(f"Parent: {parent_id} (analysis loaded, {len(parent_analysis)} chars)")
        else:
            summary_lines.append(f"Parent: {parent_id} (analysis not found)")
    else:
        summary_lines.append("Parent: None (root solution)")

    if sibling_analyses:
        summary_lines.append(f"Siblings: {len(sibling_analyses)} found")
        for sibling_data in sibling_analyses:
            summary_lines.append(f"  - {sibling_data['sibling_id']} ({len(sibling_data['analysis'])} chars)")
    else:
        summary_lines.append("Siblings: None found")

    if uncle_analyses:
        summary_lines.append(f"Uncles: {len(uncle_analyses)} found")
        for uncle_data in uncle_analyses:
            summary_lines.append(f"  - {uncle_data['uncle_id']} ({len(uncle_data['analysis'])} chars)")
    else:
        summary_lines.append("Uncles: None found")

    summary = "\n".join(summary_lines)

    return {
        "parent_id": parent_id,
        "parent_analysis": parent_analysis,
        "sibling_analyses": sibling_analyses,
        "uncle_analyses": uncle_analyses,
        "summary": summary
    }


def main():
    """Main function for testing.

    中文：命令行自测入口。解析 --champion_id（默认 solution_0），调用 retrieve_AB 并打印
        摘要，以及父/兄弟/叔伯分析报告的截断预览，便于人工核对。异常时打印错误与堆栈。
    """
    parser = argparse.ArgumentParser(description="Retrieve analysis reports (parent, siblings, uncles)")
    parser.add_argument("--champion_id", type=str, default="solution_0",
                        help="Champion solution ID (default: solution_0)")
    args = parser.parse_args()

    try:
        result = retrieve_AB(args.champion_id)

        print("\n" + "="*80)
        print("ANALYSIS REPORT RETRIEVAL (PARENT, SIBLINGS, UNCLES)")
        print("="*80)
        print(result["summary"])
        print("="*80)

        if result["parent_analysis"]:
            print("\nParent Analysis Preview:")
            print("-" * 80)
            preview = result["parent_analysis"][:500] + "..." if len(result["parent_analysis"]) > 500 else result["parent_analysis"]
            print(preview)
            print("-" * 80)

        if result["sibling_analyses"]:
            print("\nSibling Analyses:")
            for sibling_data in result["sibling_analyses"]:
                print(f"\n{sibling_data['sibling_id']}:")
                print("-" * 80)
                preview = sibling_data["analysis"][:300] + "..." if len(sibling_data["analysis"]) > 300 else sibling_data["analysis"]
                print(preview)
                print("-" * 80)

        if result["uncle_analyses"]:
            print("\nUncle Analyses:")
            for uncle_data in result["uncle_analyses"]:
                print(f"\n{uncle_data['uncle_id']}:")
                print("-" * 80)
                preview = uncle_data["analysis"][:300] + "..." if len(uncle_data["analysis"]) > 300 else uncle_data["analysis"]
                print(preview)
                print("-" * 80)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
