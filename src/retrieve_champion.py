"""
Retrieve the champion solution from results.json.

The champion is the solution with the best score that hasn't reached
the maximum children limit (10 children).
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
    """
    # Sort by score (ascending, since lower is better)
    sorted_solutions = sorted(results.items(), key=lambda x: x[1]["score"])

    # Assign ranks
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
    """
    # Sort by rank (best first)
    sorted_by_rank = sorted(results.items(), key=lambda x: x[1]["rank"])

    # Find first solution that doesn't have 10 children
    for sol_id, sol_data in sorted_by_rank:
        num_children = count_children(results, sol_id)

        if num_children < MAX_CHILDREN_PER_NODE:
            print(f"Champion selected: {sol_id} (rank {sol_data['rank']}, score {sol_data['score']:.6f}, {num_children} children)")
            return sol_id
        else:
            print(f"Skipping {sol_id} (rank {sol_data['rank']}): already has {num_children} children")

    # This should never happen unless all solutions have 10 children
    raise RuntimeError("No valid champion found - all solutions have reached max children limit!")


def retrieve_champion():
    """
    Retrieve champion solution code and analysis.

    Returns:
        Dictionary with champion_id, champion_code, champion_analysis, and champion_score
    """
    # Read results.json
    if not os.path.exists(RESULTS_FILE):
        raise FileNotFoundError(f"Results file not found: {RESULTS_FILE}")

    with open(RESULTS_FILE, 'r') as f:
        results = json.load(f)

    if not results:
        raise ValueError("Results file is empty - no solutions available")

    # Compute ranks
    results = compute_ranks(results)

    # Find champion
    champion_id = get_champion(results)
    champion_data = results[champion_id]

    # Read champion solution code (champion_id already includes "solution_" prefix)
    solution_dir = f"{SOLUTION_AND_OUTPUTS_DIR}/{champion_id}"
    solution_path = os.path.join(solution_dir, "solution.py")

    if not os.path.exists(solution_path):
        raise FileNotFoundError(f"Champion solution code not found: {solution_path}")

    with open(solution_path, 'r') as f:
        champion_code = f.read()

    # Read champion analysis
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
    """Main function for testing."""
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
