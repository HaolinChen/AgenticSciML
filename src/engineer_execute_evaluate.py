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
    """State for engineering and execution workflow"""
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
    """Create solution directory and return path"""
    solution_dir = os.path.join(SOLUTION_AND_OUTPUTS_DIR, solution_id)
    os.makedirs(solution_dir, exist_ok=True)
    return solution_dir


def save_solution_file(solution_dir: str, solution_code: str):
    """Save solution.py to solution directory"""
    solution_path = os.path.join(solution_dir, "solution.py")
    with open(solution_path, 'w') as f:
        f.write(solution_code)
    return solution_path


def copy_evaluate_to_solution_dir(solution_dir: str):
    """Copy evaluate.py to solution directory"""
    src = os.path.join(TESTING_DIR, "evaluate.py")
    dst = os.path.join(solution_dir, "evaluate.py")
    shutil.copy(src, dst)
    return dst


def copy_datasets_to_solution_dir(solution_dir: str):
    """Copy dataset files to solution directory (like evaluate.py)"""
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
    """Load existing results.json"""
    results_path = os.path.join(RESULTS_DIR, "results.json")
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            return json.load(f)
    return {}


def save_to_results_json(solution_id: str, result_data: dict):
    """Append new result to results.json with file locking to prevent race conditions"""
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
    """Load champion code, proposal, and guidelines (NO TRUNCATION)"""
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
    """Engineer agent generates solution code"""
    print("\n" + "="*80, flush=True)
    print("ENGINEER AGENT", flush=True)
    print("="*80, flush=True)

    # Determine if this is initial or refinement
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
    """Execute validation mode and evaluate"""
    print("\n" + "="*80)
    print("VALIDATION EXECUTION")
    print("="*80)

    # Execute solution in validate mode
    print(f"Running validation (timeout: {TIMEOUT_VALIDATION}s)...")
    success, train_log = execute_solution(state['solution_dir'], "validate", state['parent_id'])

    # Check if timeout but checkpoint exists
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
    """Decide next action after validation"""
    if state.get('validation_passed', False):
        return "validation_success"
    elif state.get('validation_iteration', 0) >= MAX_DEBUG_ITERATIONS:
        return "max_iterations"
    else:
        return "debug_needed"


def debug_validation_node(state: EngineerState) -> EngineerState:
    """Call debugger for validation errors"""
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
    """Execute training mode and evaluate"""
    print("\n" + "="*80)
    print("TRAINING EXECUTION")
    print("="*80)

    # Execute solution in train mode
    print(f"Running training (timeout: {TIMEOUT_TRAINING}s)...")
    success, train_log = execute_solution(state['solution_dir'], "train", state['parent_id'])

    # Check if timeout but checkpoint exists
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
    """Decide next action after training"""
    if state.get('training_passed', False):
        return "training_success"
    elif state.get('training_iteration', 0) >= MAX_DEBUG_ITERATIONS:
        return "max_iterations"
    else:
        return "debug_needed"


def debug_training_node(state: EngineerState) -> EngineerState:
    """Call debugger for training errors"""
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
    """Finalize results and save to results.json"""

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
    """Create LangGraph workflow for engineering and execution"""

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
    """

    print("\n" + "="*80, flush=True)
    print(f"ENGINEER-EXECUTE-EVALUATE: {solution_id}", flush=True)
    print("="*80, flush=True)

    # Determine parent ID from solution_id
    # E.g., "solution_00" → parent = "solution_0"
    # E.g., "solution_011" → parent = "solution_01"
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
