"""
Generate comprehensive performance analysis for a solution.

This script can be called from command line or imported by other scripts.
It runs the analyst agent on a completed solution.
"""

import argparse
import os
import json
import re
import time
import base64
import glob
from agents import analyst_agent
from constants import SOLUTION_AND_OUTPUTS_DIR, AB_DIR


def analyze_solution(solution_id: str, proposal_text: str | None = None,
                     parent_id: str | None = None):
    """
    Analyze a solution and generate comprehensive report.

    Args:
        solution_id: Solution ID (e.g., "solution_0", "solution_00")
        proposal_text: Proposal markdown content (None for root solution)
        parent_id: Parent solution ID (None for root solution)

    Returns:
        Analysis markdown text
    """
    print(f"\n{'='*80}")
    print(f"ANALYZING: {solution_id}")
    print(f"{'='*80}\n")

    # Read solution code
    solution_dir = f"{SOLUTION_AND_OUTPUTS_DIR}/{solution_id}"
    solution_path = f"{solution_dir}/solution.py"

    if not os.path.exists(solution_path):
        raise FileNotFoundError(f"Solution not found: {solution_path}")

    with open(solution_path, 'r') as f:
        solution_code = f.read()

    print(f"Loaded solution code ({len(solution_code)} chars)")

    # Read logs
    train_log_path = f"{solution_dir}/train_log.txt"
    test_log_path = f"{solution_dir}/test_log.txt"

    if not os.path.exists(train_log_path):
        raise FileNotFoundError(f"Training log not found: {train_log_path}")
    if not os.path.exists(test_log_path):
        raise FileNotFoundError(f"Testing log not found: {test_log_path}")

    with open(train_log_path, 'r') as f:
        train_log = f.read()

    with open(test_log_path, 'r') as f:
        test_log = f.read()

    print(f"Loaded train log ({len(train_log)} chars)")
    print(f"Loaded test log ({len(test_log)} chars)")

    # Check for plots in solution directory
    plot_images = []
    plot_pattern_extensions = ['*.png', '*.jpg', '*.jpeg']
    for ext in plot_pattern_extensions:
        plot_files = glob.glob(os.path.join(solution_dir, ext))
        for plot_path in plot_files:
            try:
                with open(plot_path, 'rb') as f:
                    image_bytes = f.read()
                    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
                    plot_filename = os.path.basename(plot_path)
                    plot_images.append((plot_filename, image_b64))
            except Exception as e:
                print(f"Warning: Could not load plot {plot_path}: {e}")

    if plot_images:
        print(f"Found {len(plot_images)} plot(s) for analysis")
    else:
        print("No plots found in solution directory")

    # Extract score from test_log (parse JSON at end)
    json_match = re.search(r'\{"status".*?\}', test_log)
    if json_match:
        result_json = json.loads(json_match.group())
        score = result_json.get("score", float('inf'))
        print(f"Extracted score: {score}")
    else:
        print("Warning: Could not extract score from test log, using inf")
        score = float('inf')

    # Read parent analysis if exists
    parent_analysis = None
    if parent_id:
        parent_analysis_path = f"{AB_DIR}/{parent_id}_analysis.md"
        if os.path.exists(parent_analysis_path):
            with open(parent_analysis_path, 'r') as f:
                parent_analysis = f.read()
            print(f"Loaded parent analysis ({len(parent_analysis)} chars)")
        else:
            print(f"Warning: Parent analysis not found: {parent_analysis_path}")

    # Call analyst agent
    print("\nCalling analyst agent (this may take a moment)...")
    analysis_markdown = analyst_agent(
        solution_code=solution_code,
        train_log=train_log,
        test_log=test_log,
        score=score,
        proposal=proposal_text,
        parent_analysis=parent_analysis,
        plot_images=plot_images if plot_images else None
    )

    print(f"Analysis generated ({len(analysis_markdown)} chars)")

    # Save analysis
    os.makedirs(AB_DIR, exist_ok=True)
    analysis_path = f"{AB_DIR}/{solution_id}_analysis.md"

    with open(analysis_path, 'w') as f:
        f.write(analysis_markdown)

    print(f"\n✓ Analysis saved to: {analysis_path}")

    return analysis_markdown


def main():
    """Command-line interface for analyze.py"""
    parser = argparse.ArgumentParser(description="Analyze solution performance")
    parser.add_argument("--solution_id", type=str, required=True,
                        help="Solution ID (e.g., solution_0)")
    parser.add_argument("--parent_id", type=str, default=None,
                        help="Parent solution ID for comparison")
    parser.add_argument("--proposal_file", type=str, default=None,
                        help="Path to proposal markdown file")
    args = parser.parse_args()

    # Read proposal if provided
    proposal_text = None
    if args.proposal_file:
        if os.path.exists(args.proposal_file):
            with open(args.proposal_file, 'r') as f:
                proposal_text = f.read()
            print(f"Loaded proposal ({len(proposal_text)} chars)")
        else:
            print(f"Warning: Proposal file not found: {args.proposal_file}")

    # Generate analysis
    analysis = analyze_solution(
        solution_id=args.solution_id,
        proposal_text=proposal_text,
        parent_id=args.parent_id
    )

    # Print preview
    print("\n" + "="*80)
    print("ANALYSIS PREVIEW")
    print("="*80)
    print(analysis)
    print("="*80)


if __name__ == "__main__":
    main()
