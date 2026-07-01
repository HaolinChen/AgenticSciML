"""
Retrieve the champion solution from results.json.

The champion is the solution with the best score that hasn't reached
the maximum children limit (10 children).

============================================================================
中文模块说明（学习注释）
============================================================================
作用：
    从 RESULTS/results.json 中检索当前“最优解”（champion）。champion 定义为：
    在“分数越低越好”的排名中最靠前，且尚未达到子节点数上限（MAX_CHILDREN_PER_NODE=10）
    的那个解。达到子节点上限的解会在选择时被跳过，以保证进化树能继续扩展。

在 RAG / pipeline 中的角色：
    属于检索增强记忆（RAG）组件之一，是 Phase3 进化循环里挑选“待进化父解”的入口。
    retrieve_KB（知识库检索）在未显式指定父解时，会调用本模块拿到全局 champion 作为上下文；
    进化流程也据此决定基于哪个解生成新的变异（mutation）。

主要输入：
    - RESULTS_FILE（默认 ./RESULTS/results.json）：所有历史解的元数据字典，
      每个解至少含 "score" 与 "parent_id" 字段。
    - SOLUTION_AND_OUTPUTS_DIR/{champion_id}/solution.py：champion 的源码。
    - AB_DIR/{champion_id}_analysis.md：champion 的分析报告。
    - 常量 MAX_CHILDREN_PER_NODE：每个解允许的最大子节点数。

主要输出：
    retrieve_champion() 返回字典：champion_id、champion_code、champion_analysis、
    champion_score、champion_rank。

关键函数列表：
    - compute_ranks(results)      按分数升序为每个解写入 rank 字段（越小越好）
    - count_children(results,pid) 统计某解的子节点数量
    - get_champion(results)       选出排名最靠前且未满子节点的解，返回其 id
    - retrieve_champion()         读取 results.json，选 champion 并加载其代码与分析
    - main()                      命令行自测入口，打印 champion 摘要与预览

副作用：仅读取上述文件；不调用 LLM，是纯确定性逻辑。
============================================================================
"""

import json
import os
from pathlib import Path
from constants import RESULTS_FILE, SOLUTION_AND_OUTPUTS_DIR, AB_DIR, MAX_CHILDREN_PER_NODE


def compute_ranks(results: dict) -> dict:
    """
    Add rank field to each solution based on score (lower is better).

    Args:
        results: Dictionary of solution results

    Returns:
        Updated results dictionary with rank fields

    中文：为每个解就地新增 "rank" 字段。约定“分数越低越好”，故按 score 升序排名，
        rank=1 表示最优。返回同一个（被修改后的）results 字典。仅内存操作，不写文件。
    """
    # 中文：按 score 升序排序（分数越低越好），排在最前的即最优解
    sorted_solutions = sorted(results.items(), key=lambda x: x[1]["score"])

    # 中文：从 1 开始依次赋 rank，直接写回每个解的字典
    for rank, (sol_id, _) in enumerate(sorted_solutions, start=1):
        results[sol_id]["rank"] = rank

    return results


def count_children(results: dict, parent_id: str) -> int:
    """
    Count how many children a solution has.

    Args:
        results: Dictionary of solution results
        parent_id: ID of the parent solution

    Returns:
        Number of children

    中文：统计 parent_id 作为父节点的子解数量，即 results 中 parent_id 字段等于该 id 的解个数。
        用于判断某解是否已达子节点上限。
    """
    return sum(1 for sol in results.values() if sol.get("parent_id") == parent_id)


def get_champion(results: dict) -> str:
    """
    Find the champion solution.

    The champion is the solution with the best score that hasn't reached
    the maximum children limit (10 children).

    Args:
        results: Dictionary of solution results with ranks computed

    Returns:
        Champion solution ID

    中文：从已计算 rank 的 results 中，按排名从优到劣遍历，返回第一个“子节点数未满”
        （< MAX_CHILDREN_PER_NODE）的解作为 champion。前提：results 中每个解已有 rank 字段
        （由 compute_ranks 写入）。副作用仅为打印日志。若所有解都已满员则抛 RuntimeError。
    """
    # 中文：按 rank 升序排序，rank 越小越靠前（越优）
    sorted_by_rank = sorted(results.items(), key=lambda x: x[1]["rank"])

    # 中文：从最优解开始找第一个子节点未满的解；已满的解跳过（不能再作为父解继续进化）
    for sol_id, sol_data in sorted_by_rank:
        num_children = count_children(results, sol_id)

        if num_children < MAX_CHILDREN_PER_NODE:
            print(f"Champion selected: {sol_id} (rank {sol_data['rank']}, score {sol_data['score']:.6f}, {num_children} children)")
            return sol_id
        else:
            print(f"Skipping {sol_id} (rank {sol_data['rank']}): already has {num_children} children")

    # 中文：理论上只有当所有解都达到子节点上限时才会走到这里，属异常情况
    raise RuntimeError("No valid champion found - all solutions have reached max children limit!")


def retrieve_champion():
    """
    Retrieve champion solution code and analysis.

    Returns:
        Dictionary with champion_id, champion_code, champion_analysis, and champion_score

    中文：检索流程总入口。步骤：读取 results.json → 计算 rank → 选出 champion →
        读取其 solution.py 源码与 {id}_analysis.md 分析报告，组装成字典返回。
    参数：无。
    返回：字典，含 champion_id、champion_code（源码文本）、champion_analysis（分析文本）、
        champion_score、champion_rank。
    副作用：读取 RESULTS_FILE、SOLUTION_AND_OUTPUTS_DIR 与 AB_DIR 下的相应文件；不调用 LLM。
        任一必需文件缺失会抛 FileNotFoundError，results 为空则抛 ValueError。
    """
    # 中文：读取全部历史解的元数据（results.json）
    if not os.path.exists(RESULTS_FILE):
        raise FileNotFoundError(f"Results file not found: {RESULTS_FILE}")

    with open(RESULTS_FILE, 'r') as f:
        results = json.load(f)

    if not results:
        raise ValueError("Results file is empty - no solutions available")

    # 中文：先算排名（写入 rank 字段），再据此选 champion
    results = compute_ranks(results)

    # 中文：选出当前最优且可继续进化的解
    champion_id = get_champion(results)
    champion_data = results[champion_id]

    # 中文：读取 champion 源码；champion_id 本身已含 "solution_" 前缀，目录名与之一致
    solution_dir = f"{SOLUTION_AND_OUTPUTS_DIR}/{champion_id}"
    solution_path = os.path.join(solution_dir, "solution.py")

    if not os.path.exists(solution_path):
        raise FileNotFoundError(f"Champion solution code not found: {solution_path}")

    with open(solution_path, 'r') as f:
        champion_code = f.read()

    # 中文：读取 champion 对应的分析报告（AB/ 目录下，命名约定为 {id}_analysis.md）
    analysis_path = f"{AB_DIR}/{champion_id}_analysis.md"

    if not os.path.exists(analysis_path):
        raise FileNotFoundError(f"Champion analysis not found: {analysis_path}")

    with open(analysis_path, 'r') as f:
        champion_analysis = f.read()

    return {
        "champion_id": champion_id,
        "champion_code": champion_code,
        "champion_analysis": champion_analysis,
        "champion_score": champion_data["score"],
        "champion_rank": champion_data["rank"]
    }


def main():
    """Main function for testing.

    中文：命令行自测入口。调用 retrieve_champion() 并打印 champion 的 id/rank/score，
        以及源码与分析报告的前若干字符预览，便于人工检查。异常时打印错误与堆栈。
    """
    try:
        result = retrieve_champion()

        print("\n" + "="*80)
        print("CHAMPION RETRIEVAL RESULTS")
        print("="*80)
        print(f"Champion ID: {result['champion_id']}")
        print(f"Champion Rank: {result['champion_rank']}")
        print(f"Champion Score: {result['champion_score']:.6f}")
        print(f"\nChampion Code ({len(result['champion_code'])} characters):")
        print("-" * 80)
        print(result['champion_code'][:500] + "..." if len(result['champion_code']) > 500 else result['champion_code'])
        print("-" * 80)
        print(f"\nChampion Analysis ({len(result['champion_analysis'])} characters):")
        print("-" * 80)
        print(result['champion_analysis'][:500] + "..." if len(result['champion_analysis']) > 500 else result['champion_analysis'])
        print("-" * 80)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
