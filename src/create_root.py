"""
Phase 2: Root Solution Initialization

This script generates the initial solution (solution_0) and performs validation:
1. Engineer generates solution code
2. Execute with --mode=validate (cheap, 1 epoch)
3. If fails: Validator determines tester_error or engineer_error and loops
4. If passes: Automatically proceed to full training
5. Execute with --mode=train (full training)
6. Analyst generates performance analysis
7. Store results and analysis

============================================================================
中文模块说明（学习注释）
============================================================================
作用：
    Phase2「根解生成」。生成初始解 solution_0 并跑通它：
      engineer 生成 solution.py -> validate(1 个 epoch 冒烟) -> 若失败由 validator
      判定责任方(tester_error/engineer_error)并循环修正 -> 通过后 train 全量训练 ->
      analyst 分析 -> 把结果写入 RESULTS/results.json（键 solution_0）。
    这是进化树的“根”，后续所有子代都基于它变异。

在 pipeline 中的位置：
    在 Phase1(create_contract.py) 之后运行；依赖已生成的 TESTING 合约。
    产出的 solution_0 与 results.json 是 Phase3 进化循环的起点。

主要输入：
    - 文件：USER_INPUT/problem.md、requirements.md、evaluation.md（必需）
            TESTING/guidelines.md、TESTING/evaluate.py（必需，来自 Phase1）
            USER_INPUT/dataset_config.json（可选，读取训练集信息）
    - 环境变量：constants.py 的 SCIML_* 配置（目录、模型/温度、MAX_DEBUG_ITERATIONS
      验证/调试上限、TIMEOUT_* 执行超时）；LLM API Key（.env）

主要输出：
    - 目录/文件：SOLUTION_AND_OUTPUTS/solution_0/{solution.py, evaluate.py, 数据集,
      train_log.txt, test_log.txt, MODEL_CHECKPOINT 等}
    - RESULTS/results.json 中的 solution_0 条目（score/status/是否通过/迭代次数）
    - TESTING/debugging_history.log（validator 决策历史）
    - 返回：main() 返回 0/1

关键辅助函数：
    create_solution_directory / save_solution_file / copy_evaluate_to_solution_dir /
    copy_datasets_to_solution_dir（准备解目录），log_validator_decision（记录判责历史）

LangGraph 状态图（RootState）节点与条件边（工作流走向）：
    入口 engineer(生成解) -> validate(建目录+冒烟验证+评测)
    validate 处条件边 check_validation_result：
      - "passed"        -> train（进入全量训练）
      - "failed"        -> validator（判定责任方）
      - "max_iterations"-> save（达 MAX_DEBUG_ITERATIONS 上限，直接落盘）
    validator 处条件边 route_by_culprit：
      - "tester_error"  -> tester_refine（改合约）-> engineer（重生成解）
      - "engineer_error"-> engineer（直接重生成解）
    train 处条件边 check_training_result：
      - "passed"        -> save
      - "failed"        -> engineer（带错误回工程师，训练调试循环）
      - "max_iterations"-> save
    save -> analyze -> END
    说明：request_training_approval/check_training_approval 已定义但未接入图中，
         即验证通过后“自动进入训练”，不经人审。
============================================================================
"""

from dotenv import load_dotenv
import os
import sys
import json
import shutil
import time
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END

from constants import *
from agents import (
    engineer_agent as engineer_agent_fn,
    validator_agent as validator_agent_fn,
    tester_agent as tester_agent_fn,
    execute_solution,
    execute_evaluation,
    SolutionOutput,
    ValidationDecision,
    ContractOutput
)
from analyze import analyze_solution

load_dotenv()

# ============================================================================
# State Definition
# ============================================================================

class RootState(TypedDict):
    """State for root solution workflow

    中文：根解工作流的共享状态。含用户输入(problem/requirements/evaluation/训练集信息)、
        合约文件(guidelines_md/evaluate_py)、当前解代码与目录、验证/训练的输出与是否通过、
        错误处理(error_traceback/culprit/validator_feedback)、迭代计数
        (validation_iteration/debug_iteration)、最终 score/analysis，以及未实际接线的
        human_approved（验证后自动训练，不经审批门）。
    """
    # User inputs
    problem: str
    requirements: str
    evaluation: str
    training_set_info: str  # Training dataset information (from dataset_config.json)

    # Contract files
    guidelines_md: str
    evaluate_py: str

    # Solution generation
    solution_code: str
    solution_dir: str

    # Execution and validation
    validation_output: str
    validation_passed: bool
    training_output: str
    training_passed: bool

    # Error handling
    error_traceback: str
    culprit: str  # "tester_error", "engineer_error", or "success"
    validator_feedback: str

    # Iteration tracking
    validation_iteration: int
    debug_iteration: int

    # Final results
    score: float
    analysis: str

    # Training approval (workflow bypasses approval gate - training runs automatically after validation)
    human_approved: bool


# ============================================================================
# Helper Functions
# ============================================================================

def create_solution_directory(solution_id: int = 0) -> str:
    """Create solution directory and return path

    中文：在 SOLUTION_AND_OUTPUTS 下创建 solution_{id} 目录（默认 0=根解）并返回路径。
        副作用：mkdir。
    """
    solution_dir = os.path.join(SOLUTION_AND_OUTPUTS_DIR, f"solution_{solution_id}")
    os.makedirs(solution_dir, exist_ok=True)
    return solution_dir


def save_solution_file(solution_dir: str, solution_code: str):
    """Save solution.py to solution directory

    中文：把 engineer 生成的代码写为 solution_dir/solution.py。副作用：写文件。
    """
    solution_path = os.path.join(solution_dir, "solution.py")
    with open(solution_path, 'w') as f:
        f.write(solution_code)
    return solution_path


def copy_evaluate_to_solution_dir(solution_dir: str):
    """Copy evaluate.py to solution directory

    中文：把合约里的 TESTING/evaluate.py 复制到解目录，使该解可独立执行评测。
        副作用：拷贝文件。
    """
    src = os.path.join(TESTING_DIR, "evaluate.py")
    dst = os.path.join(solution_dir, "evaluate.py")
    shutil.copy(src, dst)
    return dst


def copy_datasets_to_solution_dir(solution_dir: str):
    """Copy dataset files to solution directory (like evaluate.py)

    中文：依据 dataset_config.json 把训练集/验证集数据文件及配置本身复制进解目录，
        使解目录成为自包含的执行环境。无配置则直接返回。副作用：拷贝文件；异常仅告警。
    """
    if not os.path.exists(DATASET_CONFIG_PATH):
        return

    try:
        import json
        with open(DATASET_CONFIG_PATH, 'r') as f:
            config = json.load(f)

        # Copy training set if exists
        if "training_set" in config:
            train_file = config["training_set"]["filename"]
            src = os.path.join(USER_INPUT_DIR, train_file)
            dst = os.path.join(solution_dir, train_file)
            if os.path.exists(src):
                shutil.copy(src, dst)

        # Copy validation set if exists
        if "validation_set" in config:
            val_file = config["validation_set"]["filename"]
            src = os.path.join(USER_INPUT_DIR, val_file)
            dst = os.path.join(solution_dir, val_file)
            if os.path.exists(src):
                shutil.copy(src, dst)

        # Copy config itself
        shutil.copy(DATASET_CONFIG_PATH,
                    os.path.join(solution_dir, "dataset_config.json"))
    except Exception as e:
        print(f"Warning: Could not copy datasets: {e}")


def log_validator_decision(iteration: int, culprit: str, feedback: str, error_traceback: str):
    """
    Log validator decision to debugging history file

    Args:
        iteration: Current validation iteration number
        culprit: Who is at fault ("tester_error" or "engineer_error")
        feedback: Specific feedback from validator
        error_traceback: The error that triggered validation

    中文：把 validator 的判责结果与错误 traceback 追加写入
        TESTING/debugging_history.log，形成可追溯的调试历史。副作用：追加写文件。
    """
    log_path = os.path.join(TESTING_DIR, "debugging_history.log")

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    log_entry = f"""
{'='*80}
VALIDATION ITERATION {iteration}
Timestamp: {timestamp}
{'='*80}

CULPRIT: {culprit}

VALIDATOR FEEDBACK:
{feedback}

ERROR TRACEBACK:
{error_traceback}

{'='*80}

"""

    # Append to log file
    with open(log_path, 'a') as f:
        f.write(log_entry)

    print(f"✓ Logged validator decision to {log_path}")


# ============================================================================
# Agent Nodes
# ============================================================================

def engineer_agent(state: RootState) -> RootState:
    """
    Engineer agent generates solution code

    中文（图节点）：调用共享 engineer 智能体，依据 problem/requirements/guidelines/
        训练集信息生成 solution.py；若带有 validator_feedback 则据此在现有代码上修订。
        副作用：调用 LLM。返回：solution_code，并把 debug_iteration 加 1。
    """

    print("\n" + "="*80)
    print("ENGINEER AGENT")
    print("="*80)

    # Call shared engineer agent function
    response: SolutionOutput = engineer_agent_fn(
        problem=state['problem'],
        requirements=state['requirements'],
        guidelines=state['guidelines_md'],
        training_set_info=state.get('training_set_info', ''),
        refinement_feedback=state.get('validator_feedback', ''),
        current_solution_code=state.get('solution_code', ''),
        agent_name="root_engineer"
    )

    # Track timing and word count

    print(f"Generated solution.py ({len(response.solution_code)} chars)")

    return {
        "solution_code": response.solution_code,
        "debug_iteration": state.get('debug_iteration', 0) + 1
    }


def prepare_and_validate(state: RootState) -> RootState:
    """
    Prepare solution directory and run validation

    中文（图节点）：准备解目录并做“冒烟验证”。
        步骤：建 solution_0 目录 -> 写 solution.py -> 拷入 evaluate.py 与数据集 ->
              execute_solution(mode="validate")（1 epoch，写 train_log.txt）->
              若通过再 execute_evaluation() 打分（写 test_log.txt）。
        副作用：写文件、起子进程执行训练/评测脚本。
        返回：validation_passed 及日志/分数/error_traceback，validation_iteration+1。
        三种情况：验证+评测均过（passed）、验证过但评测失败、验证本身失败
                （后两者 passed=False，error_traceback 记录对应日志）。
    """

    print("\n" + "="*80)
    print("VALIDATION EXECUTION")
    print("="*80)

    # Create solution directory
    solution_dir = create_solution_directory(0)

    # Save solution.py
    save_solution_file(solution_dir, state['solution_code'])

    # Copy evaluate.py
    copy_evaluate_to_solution_dir(solution_dir)

    # Copy datasets if they exist
    copy_datasets_to_solution_dir(solution_dir)

    print(f"Solution directory: {solution_dir}")
    print("Running validation (--mode=validate)...")
    print(f"Monitor progress: tail -f {os.path.join(solution_dir, 'train_log.txt')}")

    # Execute validation (logs to train_log.txt)
    success, train_log = execute_solution(solution_dir, "validate")

    print(f"Validation {'PASSED' if success else 'FAILED'}")

    if success:
        # Run evaluation
        print("Running evaluation...")
        print(f"Monitor progress: tail -f {os.path.join(solution_dir, 'test_log.txt')}")

        eval_success, score, test_log = execute_evaluation(solution_dir)

        # Read log contents for state
        with open(train_log, 'r') as f:
            train_output = f.read()
        with open(test_log, 'r') as f:
            eval_output = f.read()

        if eval_success:
            print(f"✓ Validation passed! Code runs successfully.")
            print(f"  Validation score: {score}")
            return {
                "solution_dir": solution_dir,
                "validation_output": train_output + "\n\n" + eval_output,
                "validation_passed": True,
                "score": score,
                "error_traceback": "",
                "validation_iteration": state.get('validation_iteration', 0) + 1
            }
        else:
            print("Evaluation FAILED")
            return {
                "solution_dir": solution_dir,
                "validation_output": train_output + "\n\n" + eval_output,
                "validation_passed": False,
                "error_traceback": eval_output,
                "validation_iteration": state.get('validation_iteration', 0) + 1
            }
    else:
        print("Validation FAILED")
        # Read train_log for error
        with open(train_log, 'r') as f:
            train_output = f.read()

        # Create empty test_log
        test_log = os.path.join(solution_dir, "test_log.txt")
        with open(test_log, 'w') as f:
            f.write("Validation failed, evaluation did not run.\n")

        return {
            "solution_dir": solution_dir,
            "validation_output": train_output,
            "validation_passed": False,
            "error_traceback": train_output,
            "validation_iteration": state.get('validation_iteration', 0) + 1
        }


def validator_agent(state: RootState) -> RootState:
    """
    Validator agent determines culprit and provides feedback

    中文（图节点）：验证失败时，让 validator 智能体分析错误，判定责任方
        （culprit = tester_error 表示合约/评测有问题；engineer_error 表示解代码有问题），
        并给出具体修订建议。
        副作用：调用 LLM；把判责结果写入 debugging_history.log。
        返回：culprit 与 validator_feedback（供 route_by_culprit 路由使用）。
    """

    print("\n" + "="*80)
    print("VALIDATOR AGENT")
    print("="*80)

    # Call shared validator agent function
    response: ValidationDecision = validator_agent_fn(
        guidelines=state['guidelines_md'],
        solution_code=state['solution_code'],
        evaluate_code=state['evaluate_py'],
        error_traceback=state['error_traceback']
    )

    # Track timing and word count

    print(f"Culprit: {response.culprit}")
    print(f"Feedback: {response.specific_feedback}")

    # Log validator decision to debugging history
    log_validator_decision(
        iteration=state.get('validation_iteration', 0),
        culprit=response.culprit,
        feedback=response.specific_feedback,
        error_traceback=state['error_traceback']
    )

    return {
        "culprit": response.culprit,
        "validator_feedback": response.specific_feedback
    }


def tester_refine(state: RootState) -> RootState:
    """
    Tester refines contract based on validator feedback

    中文（图节点）：当判责为 tester_error 时，让 tester 依据 validator_feedback
        重新生成合约，并覆盖写回 TESTING/evaluate.py 与 guidelines.md。
        副作用：调用 LLM、覆盖写两份合约文件。
        返回：更新后的 guidelines_md/evaluate_py，并清空 validator_feedback
             （之后回到 engineer 用新合约重生成解）。
    """
    print("\n" + "="*80)
    print("TESTER REFINEMENT")
    print("="*80)

    # Call shared tester agent function with refinement feedback
    response: ContractOutput = tester_agent_fn(
        problem=state['problem'],
        requirements=state['requirements'],
        evaluation=state['evaluation'],
        refinement_feedback=state['validator_feedback']
    )

    print(f"Refined evaluate.py ({len(response.evaluate_py)} chars)")
    print(f"Refined guidelines.md ({len(response.guidelines_md)} chars)")

    # Save refined contract files
    with open(os.path.join(TESTING_DIR, "evaluate.py"), 'w') as f:
        f.write(response.evaluate_py)
    with open(os.path.join(TESTING_DIR, "guidelines.md"), 'w') as f:
        f.write(response.guidelines_md)

    return {
        "guidelines_md": response.guidelines_md,
        "evaluate_py": response.evaluate_py,
        "validator_feedback": ""  # Clear feedback for fresh start
    }


def request_training_approval(state: RootState) -> RootState:
    """
    Request human approval before expensive training

    中文（图节点，当前未接入图中）：在昂贵的全量训练前请求人工确认。
        交互式读取输入，APPROVE 则 human_approved=True，否则 False。
        注意：create_root_graph 未把该节点接线，验证通过后会“自动进入训练”，
             因此此函数目前不会被执行；保留以便需要人审时启用。
    """
    print("\n" + "="*80)
    print("VALIDATION PASSED - TRAINING APPROVAL")
    print("="*80)
    print(f"Validation score: {state['score']}")
    print("\nValidation output:")
    print("-"*80)
    print(state['validation_output'])
    print("-"*80)
    print("\nValidation passed successfully. Proceed with full training?")
    print("Type 'APPROVE' to proceed, or 'SKIP' to skip training:")
    print("(Press Ctrl+C to abort)\n")

    user_input = input("> ").strip()

    if user_input.upper() == "APPROVE":
        return {"human_approved": True}
    else:
        return {"human_approved": False}


def execute_training(state: RootState) -> RootState:
    """
    Execute full training with --mode=train

    中文（图节点）：对已通过验证的解跑全量训练。
        execute_solution(mode="train")（覆盖写 train_log.txt）-> 成功再 execute_evaluation
        对训练后模型打分（写 test_log.txt）。
        副作用：起子进程做完整训练/评测（耗时，受 TIMEOUT_TRAINING 约束）、写日志。
        返回：training_passed 与 score/日志；失败时记录 error_traceback
             （交由 check_training_result 决定是否回 engineer 调试）。
    """

    print("\n" + "="*80)
    print("FULL TRAINING EXECUTION")
    print("="*80)
    print("Running training (--mode=train)...")
    print("This may take a while...")
    print(f"Monitor progress: tail -f {os.path.join(state['solution_dir'], 'train_log.txt')}")

    # Execute training (logs to train_log.txt, overwrites validation log)
    success, train_log = execute_solution(state['solution_dir'], "train")

    print(f"Training {'PASSED' if success else 'FAILED'}")

    if success:
        # Run evaluation again on trained model
        print("Running final evaluation...")
        print(f"Monitor progress: tail -f {os.path.join(state['solution_dir'], 'test_log.txt')}")

        eval_success, score, test_log = execute_evaluation(state['solution_dir'])

        # Read log contents for state
        with open(train_log, 'r') as f:
            train_output = f.read()
        with open(test_log, 'r') as f:
            eval_output = f.read()

        if eval_success:
            print(f"Final evaluation PASSED with score: {score}")
            return {
                "training_output": train_output + "\n\n" + eval_output,
                "training_passed": True,
                "score": score
            }
        else:
            print("Final evaluation FAILED")
            return {
                "training_output": train_output + "\n\n" + eval_output,
                "training_passed": False,
                "error_traceback": eval_output
            }
    else:
        print("Training FAILED")
        # Read train_log for error
        with open(train_log, 'r') as f:
            train_output = f.read()

        # Create empty test_log
        test_log = os.path.join(state['solution_dir'], "test_log.txt")
        with open(test_log, 'w') as f:
            f.write("Training failed, final evaluation did not run.\n")

        return {
            "training_output": train_output,
            "training_passed": False,
            "error_traceback": train_output
        }




def save_results(state: RootState) -> RootState:
    """
    Save results to disk
    Note: Analysis is run automatically in the next node

    中文（图节点）：把根解结果写入 RESULTS/results.json 的 solution_0 条目
        （parent_id=None、score、status、验证/训练是否通过、验证迭代次数）。
        副作用：创建 RESULTS_DIR、写 results.json。返回：原样 state。
        紧接的 analyze 节点会自动生成分析。
    """

    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    # Create directories
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save results.json (solution_dir not stored - follows fixed naming convention)
    results = {
        "solution_0": {
            "parent_id": None,
            "score": state['score'],
            "status": "success",  # Root solution always succeeds if we reach here
            "validation_passed": state.get('validation_passed', False),
            "training_passed": state.get('training_passed', False),
            "validation_iterations": state.get('validation_iteration', 0)
        }
    }

    results_path = os.path.join(RESULTS_DIR, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Saved results to {results_path}")
    print("="*80)

    return state


def analyze_root(state: RootState) -> RootState:
    """
    Generate analysis for root solution

    中文（图节点）：调用 analyze.analyze_solution 为 solution_0 生成性能分析
        （根解无 proposal、无 parent）。副作用：调用 LLM/分析流程、可能写分析文件。
        失败不阻断工作流，仅把错误信息写入 analysis 字段。返回：analysis。
    """

    print("\n" + "="*80)
    print("ANALYSIS GENERATION")
    print("="*80)

    try:
        # Track analysis timing

        # Call analyze_solution (no proposal, no parent for root)
        analysis_markdown = analyze_solution(
            solution_id="solution_0",
            proposal_text=None,  # Root has no proposal
            parent_id=None  # Root has no parent
        )


        print(f"✓ Analysis generated ({len(analysis_markdown)} chars)")

        return {
            "analysis": analysis_markdown
        }

    except Exception as e:
        print(f"⚠ Warning: Analysis generation failed: {e}")
        # Don't fail the entire workflow if analysis fails
        return {
            "analysis": f"Analysis generation failed: {str(e)}"
        }


# ============================================================================
# Conditional Edge Logic
# ============================================================================

def check_validation_result(state: RootState) -> Literal["passed", "failed", "max_iterations"]:
    """Decide next step after validation

    中文（条件边路由，validate 之后）：
        - 通过                                    -> "passed"（去 train）
        - 未过且 validation_iteration>=上限        -> "max_iterations"（去 save 收尾）
        - 未过且未达上限                           -> "failed"（去 validator 判责）
    """
    if state['validation_passed']:
        return "passed"
    elif state.get('validation_iteration', 0) >= MAX_DEBUG_ITERATIONS:
        print(f"\n⚠ Maximum validation iterations ({MAX_DEBUG_ITERATIONS}) reached.")
        return "max_iterations"
    else:
        return "failed"


def route_by_culprit(state: RootState) -> Literal["tester_error", "engineer_error"]:
    """Route to appropriate refinement based on culprit

    中文（条件边路由，validator 之后）：按 validator 判定的责任方分流——
        "tester_error" -> 去 tester_refine 改合约；"engineer_error" -> 回 engineer 改代码。
    """
    return state['culprit']


def check_training_approval(state: RootState) -> Literal["approved", "skipped"]:
    """Check if user approved training

    中文（条件边路由，当前未接入图中）：读取 human_approved 决定 approved/skipped。
        由于审批节点未接线，此路由目前不参与实际工作流。
    """
    if state.get('human_approved', False):
        return "approved"
    else:
        return "skipped"


def check_training_result(state: RootState) -> Literal["passed", "failed", "max_iterations"]:
    """Decide next step after training

    中文（条件边路由，train 之后）：
        - 训练通过                              -> "passed"（去 save）
        - 未过且 debug_iteration>=上限           -> "max_iterations"（去 save 收尾）
        - 未过且未达上限                         -> "failed"（回 engineer 带错误调试重生成）
    """
    if state.get('training_passed', False):
        return "passed"
    elif state.get('debug_iteration', 0) >= MAX_DEBUG_ITERATIONS:
        print(f"\n⚠ Maximum debug iterations ({MAX_DEBUG_ITERATIONS}) reached.")
        return "max_iterations"
    else:
        return "failed"


# ============================================================================
# Graph Construction
# ============================================================================

def create_root_graph():
    """
    Build the LangGraph workflow for root solution initialization

    中文：装配并编译根解状态图。
        入口 engineer -> validate；validate 经 check_validation_result 分派
        train/validator/save；validator 经 route_by_culprit 分派
        tester_refine/engineer；tester_refine -> engineer；
        train 经 check_training_result 分派 save/engineer/save；save -> analyze -> END。
        注意 approval 相关节点未接线（验证通过自动训练）。返回：已编译的图。
    """
    workflow = StateGraph(RootState)

    # Add nodes
    workflow.add_node("engineer", engineer_agent)
    workflow.add_node("validate", prepare_and_validate)
    workflow.add_node("validator", validator_agent)
    workflow.add_node("tester_refine", tester_refine)
    workflow.add_node("approval", request_training_approval)
    workflow.add_node("train", execute_training)
    workflow.add_node("save", save_results)
    workflow.add_node("analyze", analyze_root)

    # Set entry point
    workflow.set_entry_point("engineer")

    # Validation loop
    workflow.add_edge("engineer", "validate")
    workflow.add_conditional_edges(
        "validate",
        check_validation_result,
        {
            "passed": "train",
            "failed": "validator",
            "max_iterations": "save"
        }
    )

    # Validator routing
    workflow.add_conditional_edges(
        "validator",
        route_by_culprit,
        {
            "tester_error": "tester_refine",
            "engineer_error": "engineer"
        }
    )

    # After tester refinement, need fresh engineer solution
    workflow.add_edge("tester_refine", "engineer")

    # Training loop
    workflow.add_conditional_edges(
        "train",
        check_training_result,
        {
            "passed": "save",
            "failed": "engineer",
            "max_iterations": "save"
        }
    )

    # Final steps
    workflow.add_edge("save", "analyze")
    workflow.add_edge("analyze", END)

    return workflow.compile()


# ============================================================================
# Main Function
# ============================================================================

def main():
    """
    Main entry point for root solution initialization

    中文（入口）：读取 USER_INPUT 三份 md 与 TESTING 合约两文件（缺失则报错退出），
        加载训练集信息，初始化 debugging_history.log，构造初始 state 并 invoke 状态图。
        副作用：读/写文件、间接触发 LLM 与子进程训练/评测。返回：0 成功 / 1 异常或中止。
    """

    print("\n" + "="*80)
    print("PHASE 2: ROOT SOLUTION INITIALIZATION")
    print("="*80 + "\n")

    # Read USER_INPUT files
    problem_path = os.path.join(USER_INPUT_DIR, "problem.md")
    requirements_path = os.path.join(USER_INPUT_DIR, "requirements.md")
    evaluation_path = os.path.join(USER_INPUT_DIR, "evaluation.md")

    if not all(os.path.exists(p) for p in [problem_path, requirements_path, evaluation_path]):
        print("ERROR: USER_INPUT files not found.")
        sys.exit(1)

    with open(problem_path, 'r') as f:
        problem = f.read()
    with open(requirements_path, 'r') as f:
        requirements = f.read()
    with open(evaluation_path, 'r') as f:
        evaluation = f.read()

    # Read TESTING contract files
    guidelines_path = os.path.join(TESTING_DIR, "guidelines.md")
    evaluate_path = os.path.join(TESTING_DIR, "evaluate.py")

    if not all(os.path.exists(p) for p in [guidelines_path, evaluate_path]):
        print("ERROR: TESTING contract files not found. Run create_contract.py first.")
        sys.exit(1)

    with open(guidelines_path, 'r') as f:
        guidelines_md = f.read()
    with open(evaluate_path, 'r') as f:
        evaluate_py = f.read()

    # Load training set info if dataset_config.json exists
    training_set_info = ""
    if os.path.exists(DATASET_CONFIG_PATH):
        try:
            with open(DATASET_CONFIG_PATH, 'r') as f:
                config = json.load(f)
            if "training_set" in config:
                training_set_info = f"""**File:** `{config['training_set']['filename']}`

**Description:** {config['training_set']['description']}

**Loading Instructions:** {config['training_set']['loading_instructions']}"""
                print(f"✓ Training dataset info loaded")
        except Exception as e:
            print(f"Warning: Could not load training dataset info: {e}")

    print("Loaded inputs:")
    print(f"  - problem.md ({len(problem)} chars)")
    print(f"  - requirements.md ({len(requirements)} chars)")
    print(f"  - evaluation.md ({len(evaluation)} chars)")
    print(f"  - guidelines.md ({len(guidelines_md)} chars)")
    print(f"  - evaluate.py ({len(evaluate_py)} chars)")
    if training_set_info:
        print(f"  - training dataset info ({len(training_set_info)} chars)")
    print()

    # Initialize debugging history log
    debug_log_path = os.path.join(TESTING_DIR, "debugging_history.log")
    with open(debug_log_path, 'w') as f:
        f.write(f"ROOT SOLUTION DEBUGGING HISTORY\n")
        f.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n\n")
    print(f"Initialized debugging log: {debug_log_path}\n")

    # Create initial state
    initial_state: RootState = {
        "problem": problem,
        "requirements": requirements,
        "evaluation": evaluation,
        "training_set_info": training_set_info,
        "guidelines_md": guidelines_md,
        "evaluate_py": evaluate_py,
        "solution_code": "",
        "solution_dir": "",
        "validation_output": "",
        "validation_passed": False,
        "training_output": "",
        "training_passed": False,
        "error_traceback": "",
        "culprit": "",
        "validator_feedback": "",
        "validation_iteration": 0,
        "debug_iteration": 0,
        "score": float('inf'),
        "analysis": "",
        "human_approved": False
    }

    # Create and run the graph
    graph = create_root_graph()

    try:
        final_state = graph.invoke(initial_state)

        # Log overall phase timing

        print("\n" + "="*80)
        print("ROOT SOLUTION INITIALIZATION COMPLETED")
        print("="*80)
        print(f"Final score: {final_state.get('score', 'N/A')}")
        print(f"Validation iterations: {final_state.get('validation_iteration', 0)}")
        print(f"Debug iterations: {final_state.get('debug_iteration', 0)}")

        # Print debugging log location if any debugging occurred
        if final_state.get('validation_iteration', 0) > 1:
            debug_log_path = os.path.join(TESTING_DIR, "debugging_history.log")
            print(f"\nDebugging history: {debug_log_path}")

        print("="*80)
        return 0

    except KeyboardInterrupt:
        print("\n✗ Root solution initialization aborted by user")
        return 1
    except Exception as e:
        print(f"\n✗ Error during root solution initialization: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
