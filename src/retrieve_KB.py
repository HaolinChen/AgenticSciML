"""
Retrieve relevant knowledge base entries for the champion solution.

Uses an LLM agent to critically evaluate KB entries and select 0-1 entry
that will help improve the champion's performance.

============================================================================
中文模块说明（学习注释）
============================================================================
作用：
    从策划好的 SciML 方法知识库（KB/，含 indices.json 索引与各方法的 .md 文档）中，
    借助 retriever 智能体（LLM）批判性地评估各条目，最多选出 1 条“最能帮助改进当前解”的
    方法条目并加载其完整 markdown 内容；也可能选 0 条（都不相关）。此外提供若干读取
    USER_INPUT 的辅助函数。

在 RAG / pipeline 中的角色：
    RAG 检索增强记忆组件之一。为“提案/工程化”阶段提供外部方法学上下文：把与当前问题、
    当前（champion 或指定父解）代码与分析最相关的一篇方法文档注入后续提示，启发新思路。

KB 目录结构约定（关键）：
    - KB_INDEX_FILE（默认 ./KB/indices.json）：条目元数据列表，每项至少含 "method_name"
      与 "filepaths"（该方法 .md 的路径，可为相对 KB_DIR 的相对路径）。
    - 各方法正文为 KB/ 下的 .md 文件，由 load_kb_entry 按元数据里的 filepaths 定位读取。

消融/开关（环境变量，关键）：
    - SCIML_KB_MODE：normal（默认，LLM 检索）/ none（完全禁用检索）/ random（随机选 1 条）。
    - SCIML_KB_RANDOM_SEED：为 random 模式设定随机种子，保证可复现。

主要输入：
    - KB_INDEX_FILE、KB_DIR 下的方法 .md。
    - USER_INPUT_DIR/{problem,requirements,evaluation}.md（问题/需求/评估描述）。
    - 参数 parent_id（可选）：不指定则自动取全局 champion 作为参照解。
    - 环境变量 SCIML_KB_MODE、SCIML_KB_RANDOM_SEED。

主要输出：
    retrieve_KB(parent_id) 返回字典：kb_entry（选中条目全文或 None）、entry_name（方法名或 None）、
    reasoning（选择理由）、summary（可读摘要）。

关键函数列表：
    - load_kb_indices()             读取 indices.json（缺失则告警并返回空列表）
    - load_problem_description()    拼接 USER_INPUT 下问题/需求/评估三份 md
    - load_kb_entry(entry_metadata) 按元数据 filepaths 定位并读取某条目全文
    - retrieve_KB(parent_id)        主流程：取参照解→载索引→按模式检索→载正文→出摘要
    - main()                        命令行自测入口

副作用：读取上述文件；normal 模式下会调用 LLM（retriever_agent）；none/random 模式不调 LLM。
============================================================================
"""

import json
import os
import argparse
import random
from constants import KB_DIR, KB_INDEX_FILE, USER_INPUT_DIR
from retrieve_champion import retrieve_champion
from agents import retriever_agent

# 中文：模块级随机数发生器，仅用于 SCIML_KB_MODE=random 消融模式随机挑选 KB 条目。
# 若设置了 SCIML_KB_RANDOM_SEED，则以该整数播种，使随机选择可复现。
_random_kb_rng = random.Random()
_random_seed = os.getenv("SCIML_KB_RANDOM_SEED")
if _random_seed is not None:
    _random_kb_rng.seed(int(_random_seed))


def load_kb_indices():
    """
    Load KB indices.json file.

    Returns:
        List of KB entry metadata dictionaries (empty list if file doesn't exist)

    中文：加载 KB 的索引文件 indices.json，返回条目元数据列表（每项含 method_name、filepaths 等）。
        文件不存在时不报错，仅打印告警并返回空列表（等效于“无知识库”）。副作用：读文件、打印日志。
    """
    if not os.path.exists(KB_INDEX_FILE):
        print(f"Warning: KB index file not found at {KB_INDEX_FILE}. Returning empty KB.")
        return []

    with open(KB_INDEX_FILE, 'r') as f:
        indices = json.load(f)

    return indices


def load_problem_description():
    """
    Load user's problem description.

    Returns:
        Combined problem, requirements, and evaluation description

    中文：从 USER_INPUT_DIR 读取并拼接“问题(problem.md)/需求(requirements.md)/评估(evaluation.md)”
        三份说明，缺哪份就跳过哪份，各段带 markdown 小标题。返回合并后的字符串（都不存在则为空串）。
        副作用：读取上述文件；不调用 LLM。
    """
    problem_file = f"{USER_INPUT_DIR}/problem.md"
    requirements_file = f"{USER_INPUT_DIR}/requirements.md"
    evaluation_file = f"{USER_INPUT_DIR}/evaluation.md"

    problem = ""
    if os.path.exists(problem_file):
        with open(problem_file, 'r') as f:
            problem += f"# Problem\n{f.read()}\n\n"

    if os.path.exists(requirements_file):
        with open(requirements_file, 'r') as f:
            problem += f"# Requirements\n{f.read()}\n\n"

    if os.path.exists(evaluation_file):
        with open(evaluation_file, 'r') as f:
            problem += f"# Evaluation Strategy\n{f.read()}\n\n"

    return problem


def load_kb_entry(entry_metadata: dict) -> str:
    """
    Load full markdown content for a KB entry.

    Args:
        entry_metadata: KB entry metadata from indices.json

    Returns:
        Full markdown content as string

    中文：根据某条目的元数据里的 filepaths 字段，定位并读取该方法的完整 markdown 正文。
        参数 entry_metadata：来自 indices.json 的单条目字典。
        返回：正文文本。副作用：读文件。
        缺少 filepaths 抛 ValueError；文件不存在抛 FileNotFoundError。
    """
    filepath = entry_metadata.get("filepaths")
    if not filepath:
        raise ValueError(f"No filepath found in KB entry: {entry_metadata}")

    # 中文：兼容“绝对/相对”两种写法——若 filepaths 不是以 KB_DIR 开头，则视为相对 KB_DIR 的路径，
    # 去掉开头的 "./" 后再拼到 KB_DIR 下；已含 KB_DIR 前缀的则原样使用。
    if not filepath.startswith(KB_DIR):
        filepath = os.path.join(KB_DIR, filepath.lstrip("./"))

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"KB entry file not found: {filepath}")

    with open(filepath, 'r') as f:
        return f.read()


def retrieve_KB(parent_id: str = None):
    """
    Retrieve 0-1 relevant knowledge base entries for the parent solution.

    Args:
        parent_id: Parent solution ID (optional, auto-detected global champion if None)

    Returns:
        Dictionary with:
        - kb_entry: Full markdown content (or None)
        - entry_name: Name of the selected entry (or None)
        - reasoning: LLM's reasoning for selection
        - summary: Human-readable summary

    中文：知识库检索主流程。
        参数 parent_id：作为参照的父解 id；为 None 时自动取全局 champion。
        返回：见上字典（kb_entry/entry_name/reasoning/summary）。
        副作用：读取解代码/分析、KB 索引与条目；normal 模式会调用 LLM，none/random 不调。
    步骤：确定参照解 → 载入 KB 索引 → 依 SCIML_KB_MODE 选择检索方式 → 载入选中条目正文 → 生成摘要。
    """
    # 中文：确定“参照解”（其代码+分析将作为检索上下文）
    if parent_id is None:
        # 中文：未指定父解 → 使用全局 champion 作为参照
        champion_data = retrieve_champion()
    else:
        # 中文：指定了父解 → 直接读取该父解的源码与分析，手工组装成与 champion 相同的结构
        print(f"Loading KB context for parent: {parent_id}")
        from constants import SOLUTION_AND_OUTPUTS_DIR, AB_DIR

        parent_code_path = os.path.join(SOLUTION_AND_OUTPUTS_DIR, parent_id, "solution.py")
        with open(parent_code_path, 'r') as f:
            parent_code = f.read()

        parent_analysis_path = os.path.join(AB_DIR, f"{parent_id}_analysis.md")
        with open(parent_analysis_path, 'r') as f:
            parent_analysis = f.read()

        champion_data = {
            "champion_id": parent_id,
            "champion_code": parent_code,
            "champion_analysis": parent_analysis
        }

    # 中文：载入 KB 索引（条目元数据列表），后续按索引下标定位/读取条目
    kb_indices = load_kb_indices()
    print(f"Loaded {len(kb_indices)} KB entries")

    # 中文：KB 消融模式（由环境变量 SCIML_KB_MODE 控制，默认 normal）：
    # - normal：默认，交给 retriever 智能体（LLM）做检索
    # - none：完全禁用知识库检索（永不选条目）
    # - random：跳过 LLM，直接随机选 1 条
    kb_mode = os.getenv("SCIML_KB_MODE", "normal").strip().lower()

    if kb_mode == "none":
        reasoning = "KB ablation mode: none. Retrieval disabled for this run."
        summary = "\n".join([
            f"Champion: {champion_data['champion_id']}",
            f"KB entries evaluated: {len(kb_indices)}",
            "Selected: None",
            f"\nReasoning:\n{reasoning}",
        ])
        return {
            "kb_entry": None,
            "entry_name": None,
            "reasoning": reasoning,
            "summary": summary
        }

    # 中文：random 模式 —— 不调用 LLM，直接随机选 1 条（用于消融对照）
    if kb_mode == "random":
        if not kb_indices:
            reasoning = "KB ablation mode: random. KB is empty, no entry selected."
            summary = "\n".join([
                f"Champion: {champion_data['champion_id']}",
                "KB entries evaluated: 0",
                "Selected: None",
                f"\nReasoning:\n{reasoning}",
            ])
            return {
                "kb_entry": None,
                "entry_name": None,
                "reasoning": reasoning,
                "summary": summary
            }

        # 中文：在 [0, 条目数) 内随机取一个下标，取该条目元数据、方法名并加载其正文
        selected_index = _random_kb_rng.randrange(len(kb_indices))
        selected_metadata = kb_indices[selected_index]
        entry_name = selected_metadata["method_name"]
        kb_entry_content = load_kb_entry(selected_metadata)
        reasoning = (
            f"KB ablation mode: random. Randomly selected index {selected_index} "
            f"out of {len(kb_indices)} entries."
        )

        summary = "\n".join([
            f"Champion: {champion_data['champion_id']}",
            f"KB entries evaluated: {len(kb_indices)}",
            f"Selected: {entry_name}",
            f"Content length: {len(kb_entry_content)} characters",
            f"\nReasoning:\n{reasoning}",
        ])
        return {
            "kb_entry": kb_entry_content,
            "entry_name": entry_name,
            "reasoning": reasoning,
            "summary": summary
        }

    # 中文：以下为 normal（默认）模式 —— 由 LLM 智能体做检索
    # 载入问题/需求/评估描述作为检索上下文
    problem_description = load_problem_description()

    # 中文：调用 retriever 智能体，输入参照解代码/分析、问题描述与 KB 索引，
    # 由其返回结构化结果 KBRetrievalResult（selected_entry_index 为选中下标或 None，含 reasoning）
    print("Calling retriever agent...")
    result = retriever_agent(
        champion_code=champion_data["champion_code"],
        champion_analysis=champion_data["champion_analysis"],
        problem_description=problem_description,
        kb_indices=kb_indices
    )

    # 中文：若智能体选中了某条目，则按其返回的下标加载对应正文；否则表示“无相关条目”
    kb_entry_content = None
    entry_name = None

    if result.selected_entry_index is not None:
        selected_metadata = kb_indices[result.selected_entry_index]
        entry_name = selected_metadata["method_name"]
        kb_entry_content = load_kb_entry(selected_metadata)
        print(f"Selected KB entry: [{result.selected_entry_index}] {entry_name}")
    else:
        print("No KB entry selected (None relevant)")

    # 中文：生成可读摘要（参照解、评估条目数、是否选中及其正文长度、以及 LLM 的选择理由）
    summary_lines = [
        f"Champion: {champion_data['champion_id']}",
        f"KB entries evaluated: {len(kb_indices)}",
    ]

    if entry_name:
        summary_lines.append(f"Selected: {entry_name}")
        summary_lines.append(f"Content length: {len(kb_entry_content)} characters")
    else:
        summary_lines.append("Selected: None")

    summary_lines.append(f"\nReasoning:\n{result.reasoning}")

    summary = "\n".join(summary_lines)

    return {
        "kb_entry": kb_entry_content,
        "entry_name": entry_name,
        "reasoning": result.reasoning,
        "summary": summary
    }


def main():
    """Main function for testing.

    中文：命令行自测入口。解析 --parent_id（默认 None 即自动取全局 champion），调用 retrieve_KB
        并打印检索摘要；若选中条目，再打印其正文前若干字符预览。异常时打印错误与堆栈。
    """
    parser = argparse.ArgumentParser(description="Retrieve relevant KB entries")
    parser.add_argument("--parent_id", type=str, default=None,
                        help="Parent solution ID (default: auto-detect global champion)")
    args = parser.parse_args()

    try:
        result = retrieve_KB(args.parent_id)

        print("\n" + "="*80)
        print("KNOWLEDGE BASE RETRIEVAL RESULTS")
        print("="*80)
        print(result["summary"])
        print("="*80)

        if result["kb_entry"]:
            print("\nKB Entry Preview (first 800 characters):")
            print("-" * 80)
            preview = result["kb_entry"][:800] + "..." if len(result["kb_entry"]) > 800 else result["kb_entry"]
            print(preview)
            print("-" * 80)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
