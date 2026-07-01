"""
Phase 1: Contract Creation

This script generates the testing contract (evaluate.py and guidelines.md)
using a Tester agent. It includes human-in-the-loop approval for refinement.

============================================================================
中文模块说明（学习注释）
============================================================================
作用：
    Phase1「测试合约生成」。由 tester 智能体读取用户需求，生成两份测试合约文件：
      - TESTING/evaluate.py    评测脚本（后续所有解都用它打分）
      - TESTING/guidelines.md  给 engineer 的工程指南（含训练集说明等）
    支持人审(HITL)：交互式下由人输入 APPROVE 通过或给出修订反馈重新生成；
    在 --mode full 批处理下由 auto_approve 自动通过（无需人工介入）。

在 pipeline 中的位置：
    总编排 src/main.py 的第一步（Phase0.5 数据分析之后、Phase2 根解生成之前）。
    本文件产出的 TESTING/evaluate.py 与 guidelines.md 是 create_root.py 与后续
    进化阶段的输入前提。

主要输入：
    - 文件：USER_INPUT/problem.md、requirements.md、evaluation.md（必需）
            USER_INPUT/dataset_config.json（可选，DATASET_CONFIG_PATH）
    - 参数：main(auto_approve)；auto_approve=True 时自动通过合约
    - 环境变量：constants.py 中的 SCIML_* 配置（如目录路径、LLM 模型/温度、
      MAX_REFINEMENT_ITERATIONS 修订上限）；以及 LLM API Key（.env 通过 load_dotenv 载入）

主要输出：
    - 文件：TESTING/evaluate.py、TESTING/guidelines.md
    - 返回：main() 返回 0（成功）/1（达到最大修订次数或异常/用户中止）

数据可见性要点：
    tester 只看到「验证集」信息（用于设计 evaluate.py）；「训练集」信息仅在
    guidelines.md 中作为文档提及，供 engineer 在后续阶段使用。

LangGraph 状态图（ContractState）节点与边（工作流走向）：
    入口 -> tester(生成合约) -> approval(展示并请求审批)
    approval 处条件边 check_approval：
      - "approved"      -> save(落盘保存) -> END
      - "refine"        -> tester（带用户反馈重新生成，循环）
      - "max_iterations"-> END（达到 MAX_REFINEMENT_ITERATIONS 上限，未保存）
    save -> END
============================================================================
"""

from dotenv import load_dotenv
import os
import sys
import time
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END

from constants import *
from agents import tester_agent as tester_agent_fn, ContractOutput
import json

load_dotenv()

# ============================================================================
# Validation Function
# ============================================================================

def validate_evaluate_py(evaluate_py_content: str) -> tuple[bool, str]:
    """Check that evaluate.py includes required JSON output format

    中文：静态检查 LLM 生成的 evaluate.py 是否含有约定的输出格式（结果标记、
        json.dumps 调用、status/score 字段）。这是执行器解析分数的契约。
    参数：evaluate_py_content 生成的 evaluate.py 源码文本。
    返回：(是否通过, 说明信息)。无副作用（不写文件、不调 LLM）。
    """
    if '--- FINAL SCALAR METRIC ---' not in evaluate_py_content:
        return False, "Missing '--- FINAL SCALAR METRIC ---' marker"

    if 'json.dumps' not in evaluate_py_content:
        return False, "Missing json.dumps() call"

    if '"status"' not in evaluate_py_content or '"score"' not in evaluate_py_content:
        return False, "Missing required JSON fields (status, score)"

    return True, "Validation passed"

# ============================================================================
# State Definition
# ============================================================================

class ContractState(TypedDict):
    """State for contract creation workflow

    中文：LangGraph 在各节点间流转的状态字典。
        problem/requirements/evaluation 为三份用户输入文本；
        evaluate_py/guidelines_md 为当前生成的合约内容；
        refinement_feedback 为人审给出的修订意见；iteration 记录已生成轮次；
        approved 是否已通过；auto_approve 是否为 --mode full 自动通过模式。
    """
    problem: str
    requirements: str
    evaluation: str

    evaluate_py: str
    guidelines_md: str

    refinement_feedback: str
    iteration: int
    approved: bool
    auto_approve: bool  # Auto-approve in --mode full (non-interactive)


# ============================================================================
# Agent Nodes
# ============================================================================

def tester_agent(state: ContractState) -> ContractState:
    """
    Tester agent node - calls shared tester_agent function from agents.py

    中文（图节点）：调用 agents.py 的共享 tester 智能体生成/修订合约。
        做什么：读取 dataset_config.json（仅取验证集信息喂给 tester，训练集信息只作
                文档提示），连同 problem/requirements/evaluation 及可能的修订反馈，
                调用 LLM 生成 evaluate.py 与 guidelines.md，并静态校验格式。
        副作用：调用 LLM；仅打印日志，不写文件（保存在 save 节点）。
        返回：更新 evaluate_py、guidelines_md，并把 iteration 加 1。
    """
    iteration = state.get('iteration', 0)

    # Load dataset info if exists
    # IMPORTANT: Tester only sees VALIDATION set (for evaluate.py design)
    # Training set info goes to Engineer via guidelines.md
    # 中文：数据可见性隔离——只把「验证集」信息拼进 dataset_info 交给 tester，
    #       训练集仅以“供 engineer 参考”的说明形式附带，避免测试端泄露训练细节。
    dataset_info = ""
    if os.path.exists(DATASET_CONFIG_PATH):
        try:
            with open(DATASET_CONFIG_PATH, 'r') as f:
                config = json.load(f)

            # ONLY pass validation set to tester
            if "validation_set" in config:
                dataset_info += "## Validation Set (For Testing)\n\n"
                dataset_info += f"**File:** `{config['validation_set']['filename']}`\n\n"
                dataset_info += f"**Description:** {config['validation_set']['description']}\n\n"
                dataset_info += f"**Loading Instructions:** {config['validation_set']['loading_instructions']}\n\n"

            # Also mention training set existence for guidelines.md documentation
            if "training_set" in config:
                dataset_info += "## Training Set (For Engineer Reference)\n\n"
                dataset_info += f"**File:** `{config['training_set']['filename']}`\n\n"
                dataset_info += f"**Description:** {config['training_set']['description']}\n\n"
                dataset_info += f"**Loading Instructions:** {config['training_set']['loading_instructions']}\n\n"
                dataset_info += "*Note: This training data is for the Engineer to use during solution development. Your evaluate.py should use the validation set above.*\n\n"

        except Exception as e:
            print(f"\nWarning: Could not load dataset_config.json: {e}")
            dataset_info = ""

    # Call shared tester agent function
    response: ContractOutput = tester_agent_fn(
        problem=state['problem'],
        requirements=state['requirements'],
        evaluation=state['evaluation'],
        dataset_info=dataset_info,
        refinement_feedback=state.get('refinement_feedback', '')
    )

    print("\n" + "="*80)
    print("TESTER AGENT OUTPUT")
    print("="*80)
    print(f"\n[Iteration {iteration + 1}]")
    print(f"\nGenerated evaluate.py ({len(response.evaluate_py)} chars)")
    print(f"Generated guidelines.md ({len(response.guidelines_md)} chars)")

    # Validate that evaluate.py has required JSON output format
    valid, message = validate_evaluate_py(response.evaluate_py)
    if not valid:
        print(f"\n⚠ WARNING: Generated evaluate.py validation failed: {message}")
        print("This will likely cause evaluation to fail.")
    else:
        print("\n✓ Generated evaluate.py passed validation")

    return {
        "evaluate_py": response.evaluate_py,
        "guidelines_md": response.guidelines_md,
        "iteration": iteration + 1
    }

def display_and_request_approval(state: ContractState) -> ContractState:
    """
    Display generated files and request human approval (or auto-approve)

    中文（图节点）：审批环节。
        - 若 auto_approve 为真（--mode full）：直接返回 approved=True，不阻塞。
        - 否则交互式：打印 evaluate.py/guidelines.md 全文，等待用户输入；
          输入 "APPROVE" 则通过，否则把输入当作修订反馈返回以触发重新生成。
        副作用：交互式分支会阻塞读取标准输入 input()；不写文件、不调 LLM。
        返回：approved（及 refine 时的 refinement_feedback）。
    """
    # Auto-approve if flag is set (used in --mode full)
    if state.get('auto_approve', False):
        print("\n" + "="*80)
        print("AUTO-APPROVE MODE (--mode full, non-interactive)")
        print("="*80)
        print("✓ Testing contract automatically approved for batch execution")
        print(f"Generated evaluate.py: {len(state['evaluate_py'])} chars")
        print(f"Generated guidelines.md: {len(state['guidelines_md'])} chars")
        return {"approved": True}

    # Interactive approval (original code for --mode contract-only)
    print("\n" + "="*80)
    print("GENERATED TESTING CONTRACT")
    print("="*80)

    print("\n" + "-"*80)
    print("FILE: evaluate.py")
    print("-"*80)
    print(state['evaluate_py'])

    print("\n" + "-"*80)
    print("FILE: guidelines.md")
    print("-"*80)
    print(state['guidelines_md'])

    print("\n" + "="*80)
    print("APPROVAL REQUEST")
    print("="*80)
    print("Please review the generated testing contract.")
    print("Type 'APPROVE' to proceed, or provide feedback for refinement:")
    print("(Press Ctrl+C to abort)\n")

    user_input = input("> ").strip()

    if user_input.upper() == "APPROVE":
        return {"approved": True}
    else:
        return {"approved": False, "refinement_feedback": user_input}

def save_contract(state: ContractState) -> ContractState:
    """
    Save the approved contract to files

    中文（图节点）：将已通过的合约落盘。
        副作用：创建 TESTING_DIR，写入 TESTING/evaluate.py 与 TESTING/guidelines.md。
        返回：原样返回 state。
    """
    os.makedirs(TESTING_DIR, exist_ok=True)

    evaluate_path = os.path.join(TESTING_DIR, "evaluate.py")
    guidelines_path = os.path.join(TESTING_DIR, "guidelines.md")

    with open(evaluate_path, 'w') as f:
        f.write(state['evaluate_py'])

    with open(guidelines_path, 'w') as f:
        f.write(state['guidelines_md'])

    print("\n" + "="*80)
    print("CONTRACT SAVED")
    print("="*80)
    print(f"✓ {evaluate_path}")
    print(f"✓ {guidelines_path}")
    print("="*80 + "\n")

    return state

# ============================================================================
# Conditional Edge Logic
# ============================================================================

def check_approval(state: ContractState) -> Literal["approved", "refine", "max_iterations"]:
    """
    Decide next step based on approval status

    中文（条件边路由）：approval 之后的分支决策。
        - approved=True                              -> "approved"（去 save 保存）
        - 已达 MAX_REFINEMENT_ITERATIONS 修订上限     -> "max_iterations"（结束，不保存）
        - 否则                                        -> "refine"（回 tester 带反馈重生成）
    """
    if state['approved']:
        return "approved"
    elif state['iteration'] >= MAX_REFINEMENT_ITERATIONS:
        print(f"\n⚠ Maximum refinement iterations ({MAX_REFINEMENT_ITERATIONS}) reached.")
        return "max_iterations"
    else:
        return "refine"

# ============================================================================
# Graph Construction
# ============================================================================

def create_contract_graph():
    """
    Build the LangGraph workflow for contract creation

    中文：装配并编译状态图。节点 tester/approval/save；入口为 tester；
        tester->approval，approval 经 check_approval 条件边分派到
        save / tester(refine) / END(max_iterations)，save->END。
    返回：已编译、可 invoke 的图对象。
    """
    workflow = StateGraph(ContractState)

    # Add nodes
    workflow.add_node("tester", tester_agent)
    workflow.add_node("approval", display_and_request_approval)
    workflow.add_node("save", save_contract)

    # Add edges
    workflow.set_entry_point("tester")
    workflow.add_edge("tester", "approval")

    # Conditional edge from approval
    workflow.add_conditional_edges(
        "approval",
        check_approval,
        {
            "approved": "save",
            "refine": "tester",
            "max_iterations": END
        }
    )

    workflow.add_edge("save", END)

    return workflow.compile()

# ============================================================================
# Main Function
# ============================================================================

def main(auto_approve=False):
    """
    Main entry point for contract creation

    Args:
        auto_approve: If True, automatically approve contract without human review
                     (set to True when called from --mode full)

    中文（入口）：读取 USER_INPUT 的三份 md（缺失则报错退出），构造初始 state，
        编译并 invoke 状态图。
        副作用：读文件、间接触发 LLM 调用与文件写入（经 save 节点）。
        返回：0 成功；1 表示达到修订上限、用户中止或异常。
    """
    print("\n" + "="*80)
    print("PHASE 1: TESTING CONTRACT CREATION")
    print("="*80 + "\n")

    # Read USER_INPUT files
    problem_path = os.path.join(USER_INPUT_DIR, "problem.md")
    requirements_path = os.path.join(USER_INPUT_DIR, "requirements.md")
    evaluation_path = os.path.join(USER_INPUT_DIR, "evaluation.md")

    if not all(os.path.exists(p) for p in [problem_path, requirements_path, evaluation_path]):
        print("ERROR: USER_INPUT files not found. Please create problem.md, requirements.md, and evaluation.md")
        sys.exit(1)

    with open(problem_path, 'r') as f:
        problem = f.read()
    with open(requirements_path, 'r') as f:
        requirements = f.read()
    with open(evaluation_path, 'r') as f:
        evaluation = f.read()

    print("Loaded USER_INPUT:")
    print(f"  - problem.md ({len(problem)} chars)")
    print(f"  - requirements.md ({len(requirements)} chars)")
    print(f"  - evaluation.md ({len(evaluation)} chars)")
    print()

    # Create initial state
    initial_state: ContractState = {
        "problem": problem,
        "requirements": requirements,
        "evaluation": evaluation,
        "evaluate_py": "",
        "guidelines_md": "",
        "refinement_feedback": "",
        "iteration": 0,
        "approved": False,
        "auto_approve": auto_approve
    }

    # Create and run the graph
    graph = create_contract_graph()

    try:
        final_state = graph.invoke(initial_state)

        if final_state['approved']:
            print("\n✓ Contract creation completed successfully!")
            return 0
        else:
            print("\n✗ Contract creation terminated (max iterations reached)")
            return 1

    except KeyboardInterrupt:
        print("\n\n✗ Contract creation aborted by user")
        return 1
    except Exception as e:
        print(f"\n✗ Error during contract creation: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
