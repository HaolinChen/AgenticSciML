"""
Phase 1: Contract Creation

This script generates the testing contract (evaluate.py and guidelines.md)
using a Tester agent. It includes human-in-the-loop approval for refinement.
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
    """Check that evaluate.py includes required JSON output format"""
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
    """State for contract creation workflow"""
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
    """
    iteration = state.get('iteration', 0)

    # Load dataset info if exists
    # IMPORTANT: Tester only sees VALIDATION set (for evaluate.py design)
    # Training set info goes to Engineer via guidelines.md
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
