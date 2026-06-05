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
    """State for root solution workflow"""
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
    """Create solution directory and return path"""
    solution_dir = os.path.join(SOLUTION_AND_OUTPUTS_DIR, f"solution_{solution_id}")
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
    """Decide next step after validation"""
    if state['validation_passed']:
        return "passed"
    elif state.get('validation_iteration', 0) >= MAX_DEBUG_ITERATIONS:
        print(f"\n⚠ Maximum validation iterations ({MAX_DEBUG_ITERATIONS}) reached.")
        return "max_iterations"
    else:
        return "failed"


def route_by_culprit(state: RootState) -> Literal["tester_error", "engineer_error"]:
    """Route to appropriate refinement based on culprit"""
    return state['culprit']


def check_training_approval(state: RootState) -> Literal["approved", "skipped"]:
    """Check if user approved training"""
    if state.get('human_approved', False):
        return "approved"
    else:
        return "skipped"


def check_training_result(state: RootState) -> Literal["passed", "failed", "max_iterations"]:
    """Decide next step after training"""
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
