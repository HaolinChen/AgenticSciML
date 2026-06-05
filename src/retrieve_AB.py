"""
Retrieve analysis reports (parent, siblings, and uncles) for the champion solution.

This is a deterministic script (no LLM) that finds analysis reports
of the champion's parent, siblings, and uncles (parent's siblings).
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
    """
    # Extract the numeric part (e.g., "solution_012" → "012")
    if not solution_id.startswith("solution_"):
        raise ValueError(f"Invalid solution ID format: {solution_id}")

    numeric_id = solution_id.replace("solution_", "")

    # Root solution has no parent
    if numeric_id == "0":
        return None

    # Compute parent ID by removing last digit
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
    """
    # Get champion's parent
    champion_data = results.get(champion_id)
    if not champion_data:
        return []

    parent_id = champion_data.get("parent_id")

    # Root solution has no siblings
    if parent_id is None:
        return []

    # Find all solutions with the same parent (excluding champion)
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
    """
    # Get champion's parent
    champion_data = results.get(champion_id)
    if not champion_data:
        return []

    parent_id = champion_data.get("parent_id")

    # Root solution has no parent, thus no uncles
    if parent_id is None:
        return []

    # Get parent's data to find grandparent
    parent_data = results.get(parent_id)
    if not parent_data:
        return []

    grandparent_id = parent_data.get("parent_id")

    # First-generation solutions (parent is root) have no uncles
    if grandparent_id is None:
        return []

    # Find all solutions with the same grandparent (parent's siblings)
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
    """
    # Read results.json to find siblings and uncles
    if not os.path.exists(RESULTS_FILE):
        raise FileNotFoundError(f"Results file not found: {RESULTS_FILE}")

    with open(RESULTS_FILE, 'r') as f:
        results = json.load(f)

    # Get parent ID from solution ID structure
    parent_id = get_parent_id(champion_id)

    # Find siblings and uncles from results.json
    sibling_ids = find_siblings(champion_id, results)
    uncle_ids = find_uncles(champion_id, results)

    # Read parent analysis
    parent_analysis = None
    if parent_id:
        parent_analysis = read_analysis(parent_id)

    # Read sibling analyses
    sibling_analyses = []
    for sibling_id in sibling_ids:
        analysis_content = read_analysis(sibling_id)
        if analysis_content:
            sibling_analyses.append({
                "sibling_id": sibling_id,
                "analysis": analysis_content
            })

    # Read uncle analyses
    uncle_analyses = []
    for uncle_id in uncle_ids:
        analysis_content = read_analysis(uncle_id)
        if analysis_content:
            uncle_analyses.append({
                "uncle_id": uncle_id,
                "analysis": analysis_content
            })

    # Generate summary
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
    """Main function for testing."""
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
