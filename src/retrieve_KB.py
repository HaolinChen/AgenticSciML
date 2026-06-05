"""
Retrieve relevant knowledge base entries for the champion solution.

Uses an LLM agent to critically evaluate KB entries and select 0-1 entry
that will help improve the champion's performance.
"""

import json
import os
import argparse
import random
from constants import KB_DIR, KB_INDEX_FILE, USER_INPUT_DIR
from retrieve_champion import retrieve_champion
from agents import retriever_agent

_random_kb_rng = random.Random()
_random_seed = os.getenv("SCIML_KB_RANDOM_SEED")
if _random_seed is not None:
    _random_kb_rng.seed(int(_random_seed))


def load_kb_indices():
    """
    Load KB indices.json file.

    Returns:
        List of KB entry metadata dictionaries (empty list if file doesn't exist)
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
    """
    filepath = entry_metadata.get("filepaths")
    if not filepath:
        raise ValueError(f"No filepath found in KB entry: {entry_metadata}")

    # Handle both absolute and relative paths
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
    """
    # Get parent data
    if parent_id is None:
        # No parent specified, use global champion
        champion_data = retrieve_champion()
    else:
        # Load specified parent's data
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

    # Load KB indices
    kb_indices = load_kb_indices()
    print(f"Loaded {len(kb_indices)} KB entries")

    # KB ablation mode:
    # - normal: default LLM-based retrieval
    # - none: disable KB completely
    # - random: pick a random KB entry directly
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

    # Load problem description
    problem_description = load_problem_description()

    # Call retriever agent
    print("Calling retriever agent...")
    result = retriever_agent(
        champion_code=champion_data["champion_code"],
        champion_analysis=champion_data["champion_analysis"],
        problem_description=problem_description,
        kb_indices=kb_indices
    )

    # Load KB entry if one was selected
    kb_entry_content = None
    entry_name = None

    if result.selected_entry_index is not None:
        selected_metadata = kb_indices[result.selected_entry_index]
        entry_name = selected_metadata["method_name"]
        kb_entry_content = load_kb_entry(selected_metadata)
        print(f"Selected KB entry: [{result.selected_entry_index}] {entry_name}")
    else:
        print("No KB entry selected (None relevant)")

    # Generate summary
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
    """Main function for testing."""
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
