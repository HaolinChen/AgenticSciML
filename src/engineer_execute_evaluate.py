"""
Phase 3B Session 3: Engineering and Execution

This script handles child solution engineering, validation, training, and evaluation:
1. Load champion code, proposal, and testing contract
2. Engineer generates solution code
3. Validation loop with debugger (max 5 iterations)
4. Training loop with debugger (max 5 iterations)
5. Store results in results.json

NO human approval gates - fully automated for children.
Timeout is NOT an error - partial checkpoints are evaluated and stored.

============================================================================
中文模块说明（学习注释）
============================================================================
作用：
    Phase3「子代工程化引擎」。给定一个 solution_id，围绕其父代冠军代码(champion)与
    某条进化提案(proposal)，生成 -> 验证 -> 训练 -> 评测该子代解，并把结果写入
    results.json。全程无人工审批，供进化循环并行批量调用。也被根解路径间接复用同类
    执行/评测原语（execute_solution/execute_evaluation）。

在 pipeline 中的位置：
    Phase3 进化循环中，每产生一个新的 solution_id 就调用本模块跑通它。上游依赖：
    父代 solution.py（champion）、PROPOSAL_POOL 里的 proposal、TESTING 合约。
    下游：results.json 中新增该 solution_id 的记录，供选择/变异使用。

engineer-execute-debug 循环（核心）：
    engineer 生成/修订 solution.py -> validate（冒烟）-> 若失败调用 debugger 产出修复
    建议并回到 engineer（validation_iteration+1），最多 MAX_DEBUG_ITERATIONS 次；验证通过
    后进入 train（全量）-> 若失败同样经 debugger 回 engineer（training_iteration+1），最多
    MAX_DEBUG_ITERATIONS 次。每次 debugger 建议会随错误信息拼进下一轮 engineer 的
    refinement_feedback。达上限或成功后进入 finalize 落盘。
    超时特例：validate/train 超时但已生成 MODEL_CHECKPOINT，则视作“非错误”，继续用部分
    检查点做评测并存分数，不触发调试。

主要输入：
    - 参数：solution_id（如 "solution_00"，其父代由去掉末位推得："solution_0"）；
            gpu_id（被忽略，实际由 main.py 通过 CUDA_VISIBLE_DEVICES 指定）
    - 文件：父代 SOLUTION_AND_OUTPUTS/{parent_id}/solution.py（champion）
            PROPOSAL_POOL/proposal_{id}.md、TESTING/guidelines.md、
            USER_INPUT/problem.md、requirements.md、可选 dataset_config.json
    - 环境变量：constants.py 的 SCIML_*（目录、模型/温度、MAX_DEBUG_ITERATIONS、
      TIMEOUT_VALIDATION/TIMEOUT_TRAINING）；LLM API Key（.env）

主要输出：
    - 目录/文件：SOLUTION_AND_OUTPUTS/{solution_id}/{solution.py, evaluate.py, 数据集,
      train_log.txt, test_log.txt, MODEL_CHECKPOINT 等}
    - RESULTS/results.json 中新增 {solution_id: {parent_id, score, status,
      validation_passed, training_passed, validation_iterations, training_iterations}}
      （带文件锁写入，避免多进程竞争）
    - 返回值：engineer_execute_evaluate() 返回上述结果的字典（额外含 solution_id/parent_id）；
      CLI 方式则把该字典以 JSON 打印。status ∈ {success, validation_failed, training_failed}

LangGraph 状态图（EngineerState）节点与条件边：
    入口 load_inputs -> engineer -> validate
    validate 经 check_validation_node：
      - "validation_success" -> train
      - "debug_needed"       -> debug_validation -> engineer（验证调试循环）
      - "max_iterations"     -> finalize
    train 经 check_training_node：
      - "training_success"   -> finalize
      - "debug_needed"       -> debug_training -> engineer（训练调试循环）
      - "max_iterations"     -> finalize
    finalize -> END
============================================================================
"""

from dotenv import load_dotenv
import os
import json
import shutil
import argparse
import time
from typing import TypedDict
from langgraph.graph import StateGraph, END

from constants import *
from agents import (
    engineer_agent as engineer_agent_fn,
    debugger_agent as debugger_agent_fn,
    execute_solution,
    execute_evaluation,
    SolutionOutput,
    DebugSuggestion
)

load_dotenv()

# ============================================================================
# State Definition
# ============================================================================

class EngineerState(TypedDict):
    """State for engineering and execution workflow

    中文：子代工程化工作流的共享状态。含标识(solution_id/parent_id)、输入(父代
        champion_code、proposal、guidelines、problem、requirements、训练集信息，均不截断)、
        当前解代码与目录、验证/训练是否通过与各自迭代计数、错误处理
        (error_message/error_traceback/debugger_suggestion)、最终 score 与 status。
    """
    # Identifiers
    solution_id: str
    parent_id: str

    # Inputs (NO TRUNCATION - full content)
    champion_code: str
    proposal: str
    guidelines: str
    problem: str
    requirements: str
    training_set_info: str  # Training dataset information (from dataset_config.json)

    # Solution generation
    solution_code: str
    solution_dir: str

    # Validation tracking
    validation_passed: bool
    validation_iteration: int

    # Training tracking
    training_passed: bool
    training_iteration: int

    # Error handling
    error_message: str
    error_traceback: str
    debugger_suggestion: str

    # Final results
    score: float
    status: str  # "success", "validation_failed", "training_failed"


# ============================================================================
# Helper Functions
# ============================================================================

def create_solution_directory(solution_id: str) -> str:
    """Create solution directory and return path

    中文：在 SOLUTION_AND_OUTPUTS 下创建该子代的目录并返回路径。副作用：mkdir。
    """
    solution_dir = os.path.join(SOLUTION_AND_OUTPUTS_DIR, solution_id)
    os.makedirs(solution_dir, exist_ok=True)
    return solution_dir


def save_solution_file(solution_dir: str, solution_code: str):
    """Save solution.py to solution directory

    中文：把生成的代码写为 solution_dir/solution.py。副作用：写文件。
    """
    solution_path = os.path.join(solution_dir, "solution.py")
    with open(solution_path, 'w') as f:
        f.write(solution_code)
    return solution_path


def copy_evaluate_to_solution_dir(solution_dir: str):
    """Copy evaluate.py to solution directory

    中文：把合约 TESTING/evaluate.py 复制进解目录，使其可独立评测。副作用：拷贝文件。
    """
    src = os.path.join(TESTING_DIR, "evaluate.py")
    dst = os.path.join(solution_dir, "evaluate.py")
    shutil.copy(src, dst)
    return dst


def copy_datasets_to_solution_dir(solution_dir: str):
    """Copy dataset files to solution directory (like evaluate.py)

    中文：依据 dataset_config.json 把训练/验证数据文件及配置复制进解目录，
        使其成为自包含执行环境。无配置则返回；异常仅告警。副作用：拷贝文件。
    """
    if not os.path.exists(DATASET_CONFIG_PATH):
        return

    try:
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


def load_results_json() -> dict:
    """Load existing results.json

    中文：读取 RESULTS/results.json（不存在返回空字典）。无副作用。
    """
    results_path = os.path.join(RESULTS_DIR, "results.json")
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            return json.load(f)
    return {}


def save_to_results_json(solution_id: str, result_data: dict):
    """Append new result to results.json with file locking to prevent race conditions

    中文：把某子代结果写入 results.json[solution_id]。因进化循环会并行多进程写同一文件，
        这里用「独占创建 .lock 文件」做互斥锁：抢到锁后读-改-写，最后必删锁；抢不到则
        重试（最多 max_retries 次，每次 retry_delay 秒），超时抛 RuntimeError。
        副作用：写 results.json、临时创建/删除 .lock 文件。
    """
    import time

    results_path = os.path.join(RESULTS_DIR, "results.json")
    lock_path = results_path + ".lock"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    max_retries = 50  # 5 seconds max wait
    retry_delay = 0.1  # 100ms between retries

    for attempt in range(max_retries):
        try:
            # Try to create lock file exclusively (atomic operation)
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            try:
                # We have the lock - perform read-modify-write
                results = load_results_json()
                results[solution_id] = result_data

                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=2)

                print(f"[{solution_id}] Saved to results.json")
            finally:
                # Always release lock
                os.close(lock_fd)
                try:
                    os.remove(lock_path)
                except OSError:
                    pass  # Lock file already removed, ignore
            break  # Success - exit retry loop

        except FileExistsError:
            # Lock file exists - another process has the lock
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise RuntimeError(
                    f"Could not acquire lock for results.json after {max_retries * retry_delay:.1f}s. "
                    f"Lock file: {lock_path}"
                )
        except Exception as e:
            # Unexpected error - clean up lock and re-raise
            try:
                if 'lock_fd' in locals():
                    os.close(lock_fd)
                os.remove(lock_path)
            except:
                pass
            raise


# ============================================================================
# Workflow Nodes
# ============================================================================

def load_inputs_node(state: EngineerState) -> EngineerState:
    """Load champion code, proposal, and guidelines (NO TRUNCATION)

    中文（图节点，入口）：加载本子代所需全部上下文（均不截断）：
        父代 solution.py（champion_code）、PROPOSAL_POOL/proposal_{id}.md、
        TESTING/guidelines.md、USER_INPUT/problem.md 与 requirements.md、可选训练集信息。
        副作用：读多份文件。返回：把这些内容填入 state。
    """
    print("\n" + "="*80, flush=True)
    print(f"LOADING INPUTS FOR {state['solution_id']}", flush=True)
    print("="*80, flush=True)

    # Load champion code (FULL, no truncation)
    champion_id = state['parent_id']
    champion_dir = os.path.join(SOLUTION_AND_OUTPUTS_DIR, champion_id)
    champion_code_path = os.path.join(champion_dir, "solution.py")
    with open(champion_code_path, 'r') as f:
        champion_code = f.read()

    # Load proposal (FULL, no truncation)
    proposal_id = state['solution_id'].replace('solution_', '')
    proposal_path = os.path.join(PROPOSAL_POOL_DIR, f"proposal_{proposal_id}.md")
    with open(proposal_path, 'r') as f:
        proposal = f.read()

    # Load guidelines
    guidelines_path = os.path.join(TESTING_DIR, "guidelines.md")
    with open(guidelines_path, 'r') as f:
        guidelines = f.read()

    # Load problem description
    problem_path = os.path.join(USER_INPUT_DIR, "problem.md")
    with open(problem_path, 'r') as f:
        problem = f.read()

    # Load requirements
    requirements_path = os.path.join(USER_INPUT_DIR, "requirements.md")
    with open(requirements_path, 'r') as f:
        requirements = f.read()

    # Load training set info if exists
    training_set_info = ""
    if os.path.exists(DATASET_CONFIG_PATH):
        try:
            with open(DATASET_CONFIG_PATH, 'r') as f:
                config = json.load(f)
            if "training_set" in config:
                training_set_info = f"""**File:** `{config['training_set']['filename']}`

**Description:** {config['training_set']['description']}

**Loading Instructions:** {config['training_set']['loading_instructions']}"""
                print(f"✓ Loaded training dataset info: {len(training_set_info)} chars", flush=True)
        except Exception as e:
            print(f"Warning: Could not load training dataset info: {e}", flush=True)

    print(f"✓ Loaded champion code: {len(champion_code)} chars (NO TRUNCATION)", flush=True)
    print(f"✓ Loaded proposal: {len(proposal)} chars (NO TRUNCATION)", flush=True)
    print(f"✓ Loaded guidelines: {len(guidelines)} chars", flush=True)
    print("✓ Returning from load_inputs_node", flush=True)

    return {
        "champion_code": champion_code,
        "proposal": proposal,
        "guidelines": guidelines,
        "problem": problem,
        "requirements": requirements,
        "training_set_info": training_set_info
    }


def engineer_node(state: EngineerState) -> EngineerState:
    """Engineer agent generates solution code

    中文（图节点，engineer-execute-debug 循环的“工程”环节）：
        依据 champion_code + proposal + guidelines 生成 solution.py；若已有
        debugger_suggestion（说明是循环中的修订），则把调试建议+错误信息拼成
        refinement_feedback 传入，让 LLM 在现有代码上修复。
        副作用：调用 LLM；首次会建解目录；每轮都写 solution.py 并拷入 evaluate.py 与数据集。
        返回：solution_code 与 solution_dir（务必回传 solution_dir 以更新 state）。
    """
    print("\n" + "="*80, flush=True)
    print("ENGINEER AGENT", flush=True)
    print("="*80, flush=True)

    # Determine if this is initial or refinement
    # 中文：有 debugger_suggestion 即表示处于调试循环中（非首次生成）。
    is_refinement = bool(state.get('debugger_suggestion', ''))

    # Build refinement feedback
    if is_refinement:
        refinement_feedback = f"""DEBUGGING SUGGESTION:
{state['debugger_suggestion']}

ERROR MESSAGE:
{state.get('error_message', '')}

ERROR TRACEBACK:
{state.get('error_traceback', '')}
"""
    else:
        refinement_feedback = ""

    # Call engineer agent (FULL champion code and proposal, NO TRUNCATION)
    print("Calling engineer_agent_fn (LLM call may take 1-2 minutes)...", flush=True)
    response: SolutionOutput = engineer_agent_fn(
        problem=state['problem'],
        requirements=state['requirements'],
        guidelines=state['guidelines'],
        training_set_info=state.get('training_set_info', ''),
        refinement_feedback=refinement_feedback,
        champion_code=state.get('champion_code', ''),
        proposal=state.get('proposal', ''),
        current_solution_code=state.get('solution_code', '')
    )

    print(f"Generated solution.py ({len(response.solution_code)} chars)", flush=True)

    # Create solution directory if not exists
    if not state.get('solution_dir'):
        solution_dir = create_solution_directory(state['solution_id'])
        state['solution_dir'] = solution_dir
        print(f"✓ Created directory: {solution_dir}")

    # Save solution.py
    save_solution_file(state['solution_dir'], response.solution_code)

    # Copy evaluate.py
    copy_evaluate_to_solution_dir(state['solution_dir'])

    # Copy datasets if they exist
    copy_datasets_to_solution_dir(state['solution_dir'])

    return {
        "solution_code": response.solution_code,
        "solution_dir": state['solution_dir']  # IMPORTANT: Return solution_dir to update state
    }


def validate_node(state: EngineerState) -> EngineerState:
    """Execute validation mode and evaluate

    中文（图节点，“执行”环节-冒烟验证）：
        execute_solution(mode="validate") 跑 1 epoch 冒烟；成功（或“超时但有检查点”）
        则 execute_evaluation 打分。副作用：起子进程执行、写 train_log/test_log。
        返回：validation_passed 及 score 或 error_message/error_traceback（供调试判定）。
    """
    print("\n" + "="*80)
    print("VALIDATION EXECUTION")
    print("="*80)

    # Execute solution in validate mode
    print(f"Running validation (timeout: {TIMEOUT_VALIDATION}s)...")
    success, train_log = execute_solution(state['solution_dir'], "validate", state['parent_id'])

    # Check if timeout but checkpoint exists
    # 中文：超时特例——虽未成功但已产出 MODEL_CHECKPOINT，则视为“非错误”，仍继续评测。
    checkpoint_path = os.path.join(state['solution_dir'], "MODEL_CHECKPOINT")
    timeout_with_checkpoint = (not success) and os.path.exists(checkpoint_path)

    if timeout_with_checkpoint:
        print("⚠ Validation timed out, but checkpoint exists - continuing to evaluation")

    # Run evaluation (even if timeout but checkpoint exists)
    if success or timeout_with_checkpoint:
        print("Running evaluation...")
        eval_success, score, test_log = execute_evaluation(state['solution_dir'])

        if eval_success:
            print(f"✓ Validation passed! Score: {score}")
            return {
                "validation_passed": True,
                "score": score,
                "error_message": "",
                "error_traceback": ""
            }
        else:
            # Evaluation failed - read error from test log
            with open(test_log, 'r') as f:
                error_output = f.read()
            print(f"✗ Evaluation failed")
            return {
                "validation_passed": False,
                "error_message": "Evaluation failed",
                "error_traceback": error_output
            }
    else:
        # Training failed - read error from train log
        with open(train_log, 'r') as f:
            error_output = f.read()
        print(f"✗ Validation failed")
        return {
            "validation_passed": False,
            "error_message": "Validation execution failed",
            "error_traceback": error_output
        }


def check_validation_node(state: EngineerState) -> str:
    """Decide next action after validation

    中文（条件边路由，validate 之后）：
        - 通过                                   -> "validation_success"（去 train）
        - 未过且 validation_iteration>=上限       -> "max_iterations"（去 finalize）
        - 未过且未达上限                          -> "debug_needed"（去 debug_validation）
    """
    if state.get('validation_passed', False):
        return "validation_success"
    elif state.get('validation_iteration', 0) >= MAX_DEBUG_ITERATIONS:
        return "max_iterations"
    else:
        return "debug_needed"


def debug_validation_node(state: EngineerState) -> EngineerState:
    """Call debugger for validation errors

    中文（图节点，“调试”环节）：验证失败时调用 debugger 智能体，依据错误与代码给出
        修复建议。副作用：调用 LLM。返回：debugger_suggestion 与 validation_iteration+1，
        随后回到 engineer 重生成（构成验证调试循环）。
    """
    print("\n" + "="*80)
    print(f"DEBUGGER (Validation Iteration {state.get('validation_iteration', 0) + 1}/{MAX_DEBUG_ITERATIONS})")
    print("="*80)

    # Call debugger agent
    suggestion: DebugSuggestion = debugger_agent_fn(
        error_message=state.get('error_message', ''),
        error_traceback=state.get('error_traceback', ''),
        solution_code=state.get('solution_code', ''),
        problem=state.get('problem', ''),
        requirements=state.get('requirements', ''),
        guidelines=state.get('guidelines', '')
    )

    print(f"Debugger suggestion: {suggestion.suggestion}")

    return {
        "debugger_suggestion": suggestion.suggestion,
        "validation_iteration": state.get('validation_iteration', 0) + 1
    }


def train_node(state: EngineerState) -> EngineerState:
    """Execute training mode and evaluate

    中文（图节点，“执行”环节-全量训练）：验证通过后跑全量训练并评测。
        超时但有检查点同样视为“非错误”，用部分检查点评测并存分数。
        副作用：起子进程执行（耗时，受 TIMEOUT_TRAINING 约束）、写 train_log/test_log。
        返回：training_passed 及 score 或 error_message/error_traceback。
    """
    print("\n" + "="*80)
    print("TRAINING EXECUTION")
    print("="*80)

    # Execute solution in train mode
    print(f"Running training (timeout: {TIMEOUT_TRAINING}s)...")
    success, train_log = execute_solution(state['solution_dir'], "train", state['parent_id'])

    # Check if timeout but checkpoint exists
    # 中文：训练超时但已有检查点属正常情况，用部分结果继续评测（见模块头“超时特例”）。
    checkpoint_path = os.path.join(state['solution_dir'], "MODEL_CHECKPOINT")
    timeout_with_checkpoint = (not success) and os.path.exists(checkpoint_path)

    if timeout_with_checkpoint:
        print("⚠ Training timed out, but checkpoint exists - continuing to evaluation")
        print("  (This is NOT an error - partial results will be stored)")

    # Run evaluation (even if timeout but checkpoint exists)
    if success or timeout_with_checkpoint:
        print("Running evaluation...")
        eval_success, score, test_log = execute_evaluation(state['solution_dir'])

        if eval_success:
            print(f"✓ Training completed! Score: {score}")
            return {
                "training_passed": True,
                "score": score,
                "error_message": "",
                "error_traceback": ""
            }
        else:
            # Evaluation failed - read error from test log
            with open(test_log, 'r') as f:
                error_output = f.read()
            print(f"✗ Evaluation failed")
            return {
                "training_passed": False,
                "error_message": "Evaluation failed after training",
                "error_traceback": error_output
            }
    else:
        # Training failed - read error from train log
        with open(train_log, 'r') as f:
            error_output = f.read()
        print(f"✗ Training failed")
        return {
            "training_passed": False,
            "error_message": "Training execution failed",
            "error_traceback": error_output
        }


def check_training_node(state: EngineerState) -> str:
    """Decide next action after training

    中文（条件边路由，train 之后）：
        - 通过                                 -> "training_success"（去 finalize）
        - 未过且 training_iteration>=上限       -> "max_iterations"（去 finalize）
        - 未过且未达上限                        -> "debug_needed"（去 debug_training）
    """
    if state.get('training_passed', False):
        return "training_success"
    elif state.get('training_iteration', 0) >= MAX_DEBUG_ITERATIONS:
        return "max_iterations"
    else:
        return "debug_needed"


def debug_training_node(state: EngineerState) -> EngineerState:
    """Call debugger for training errors

    中文（图节点，“调试”环节）：训练失败时调用 debugger 给出修复建议。副作用：调用 LLM。
        返回：debugger_suggestion 与 training_iteration+1，随后回 engineer 重生成
        （构成训练调试循环）。
    """
    print("\n" + "="*80)
    print(f"DEBUGGER (Training Iteration {state.get('training_iteration', 0) + 1}/{MAX_DEBUG_ITERATIONS})")
    print("="*80)

    # Call debugger agent
    suggestion: DebugSuggestion = debugger_agent_fn(
        error_message=state.get('error_message', ''),
        error_traceback=state.get('error_traceback', ''),
        solution_code=state.get('solution_code', ''),
        problem=state.get('problem', ''),
        requirements=state.get('requirements', ''),
        guidelines=state.get('guidelines', '')
    )

    print(f"Debugger suggestion: {suggestion.suggestion}")

    return {
        "debugger_suggestion": suggestion.suggestion,
        "training_iteration": state.get('training_iteration', 0) + 1
    }


def finalize_node(state: EngineerState) -> EngineerState:
    """Finalize results and save to results.json

    中文（图节点，收尾）：据验证/训练是否通过判定 status（success / validation_failed /
        training_failed），组装结果并经带锁的 save_to_results_json 写入 results.json。
        副作用：写 results.json。返回：status。
    """

    print("\n" + "="*80)
    print("FINALIZING RESULTS")
    print("="*80)

    # Determine status
    if state.get('validation_passed') and state.get('training_passed'):
        status = "success"
    elif not state.get('validation_passed'):
        status = "validation_failed"
    else:
        status = "training_failed"

    # Prepare results (solution_dir not stored - it follows fixed naming convention)
    result_data = {
        "parent_id": state['parent_id'],
        "score": state.get('score', float('inf')),
        "status": status,
        "validation_passed": state.get('validation_passed', False),
        "training_passed": state.get('training_passed', False),
        "validation_iterations": state.get('validation_iteration', 0),
        "training_iterations": state.get('training_iteration', 0)
    }

    # Save to results.json
    save_to_results_json(state['solution_id'], result_data)

    print(f"✓ Status: {status}")
    print(f"✓ Score: {result_data['score']}")
    print(f"✓ Saved results to results.json")
    print("="*80)

    return {
        "status": status
    }


# ============================================================================
# Workflow Construction
# ============================================================================

def create_engineer_workflow():
    """Create LangGraph workflow for engineering and execution

    中文：装配并编译子代工程化状态图。
        入口 load_inputs -> engineer -> validate；validate 经 check_validation_node
        分派 train / debug_validation / finalize；debug_validation -> engineer；
        train 经 check_training_node 分派 finalize / debug_training / finalize；
        debug_training -> engineer；finalize -> END。返回：已编译的图。
    """

    workflow = StateGraph(EngineerState)

    # Add nodes
    workflow.add_node("load_inputs", load_inputs_node)
    workflow.add_node("engineer", engineer_node)
    workflow.add_node("validate", validate_node)
    workflow.add_node("debug_validation", debug_validation_node)
    workflow.add_node("train", train_node)
    workflow.add_node("debug_training", debug_training_node)
    workflow.add_node("finalize", finalize_node)

    # Set entry point
    workflow.set_entry_point("load_inputs")

    # Linear edges
    workflow.add_edge("load_inputs", "engineer")
    workflow.add_edge("engineer", "validate")

    # Conditional edges after validation
    workflow.add_conditional_edges(
        "validate",
        check_validation_node,
        {
            "validation_success": "train",
            "debug_needed": "debug_validation",
            "max_iterations": "finalize"
        }
    )

    # Debug loop for validation
    workflow.add_edge("debug_validation", "engineer")

    # Conditional edges after training
    workflow.add_conditional_edges(
        "train",
        check_training_node,
        {
            "training_success": "finalize",
            "debug_needed": "debug_training",
            "max_iterations": "finalize"
        }
    )

    # Debug loop for training
    workflow.add_edge("debug_training", "engineer")

    # End
    workflow.add_edge("finalize", END)

    return workflow.compile()


# ============================================================================
# Main Function
# ============================================================================

def engineer_execute_evaluate(solution_id: str, gpu_id: int | None = None) -> dict:
    """
    Engineer, execute, and evaluate a child solution

    Args:
        solution_id: Solution ID (e.g., "solution_00")
        gpu_id: GPU ID (ignored - main.py sets CUDA_VISIBLE_DEVICES)

    Returns:
        Dictionary with results:
        - solution_id, parent_id, score, status
        - validation_passed, training_passed
        - validation_iterations, training_iterations
        - solution_dir

    中文（对外主入口）：本模块的编程接口，供进化循环/CLI 调用。
        由 solution_id 推导 parent_id（去掉末位数字；solution_0 为根解，禁止在此运行），
        初始化 state，编译并 invoke 状态图，最后返回结果字典。
        副作用：经各节点读写文件、调用 LLM、起子进程训练/评测、写 results.json。
    """

    print("\n" + "="*80, flush=True)
    print(f"ENGINEER-EXECUTE-EVALUATE: {solution_id}", flush=True)
    print("="*80, flush=True)

    # Determine parent ID from solution_id
    # E.g., "solution_00" → parent = "solution_0"
    # E.g., "solution_011" → parent = "solution_01"
    # 中文：进化树用“末位追加”编码父子关系，故去掉末位数字即得父代 id。
    numeric_id = solution_id.replace('solution_', '')

    if numeric_id == "0":
        raise ValueError(f"Cannot run engineer_execute_evaluate on root solution (solution_0). Root should be created via create_root.py")

    parent_numeric_id = numeric_id[:-1]  # Remove last digit
    parent_id = f"solution_{parent_numeric_id}"

    # Initialize state
    initial_state = EngineerState(
        solution_id=solution_id,
        parent_id=parent_id,
        champion_code="",  # Loaded in load_inputs_node
        proposal="",  # Loaded in load_inputs_node
        guidelines="",  # Loaded in load_inputs_node
        problem="",  # Loaded in load_inputs_node
        requirements="",  # Loaded in load_inputs_node
        training_set_info="",  # Loaded in load_inputs_node
        solution_code="",
        solution_dir="",
        validation_passed=False,
        validation_iteration=0,
        training_passed=False,
        training_iteration=0,
        error_message="",
        error_traceback="",
        debugger_suggestion="",
        score=float('inf'),
        status="unknown"
    )

    # Run workflow
    print(f"Creating workflow...", flush=True)
    workflow = create_engineer_workflow()
    print(f"Invoking workflow...", flush=True)
    final_state = workflow.invoke(initial_state)
    print(f"Workflow completed", flush=True)

    # Track overall timing

    # Return results
    return {
        "solution_id": solution_id,
        "parent_id": parent_id,
        "score": final_state.get('score', float('inf')),
        "status": final_state.get('status', 'unknown'),
        "validation_passed": final_state.get('validation_passed', False),
        "training_passed": final_state.get('training_passed', False),
        "validation_iterations": final_state.get('validation_iteration', 0),
        "training_iterations": final_state.get('training_iteration', 0)
    }


# ============================================================================
# CLI Interface
# ============================================================================

if __name__ == "__main__":
    # 中文（CLI 入口）：解析 --solution_id（必填）与 --gpu_id（被忽略），
    #   调用 engineer_execute_evaluate 跑完整流程，并把结果字典以 JSON 打印。
    print("Starting engineer_execute_evaluate.py...", flush=True)
    parser = argparse.ArgumentParser(description="Engineer, execute, and evaluate a child solution")
    parser.add_argument("--solution_id", type=str, required=True, help="Solution ID (e.g., solution_00)")
    parser.add_argument("--gpu_id", type=int, default=None, help="GPU ID (ignored - set CUDA_VISIBLE_DEVICES)")
    args = parser.parse_args()
    print(f"Parsed args: solution_id={args.solution_id}", flush=True)

    results = engineer_execute_evaluate(args.solution_id, args.gpu_id)

    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(json.dumps(results, indent=2))
    print("="*80)
