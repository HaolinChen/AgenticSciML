"""
Main orchestrator for the SciML Agent evolutionary system.

This script coordinates the full workflow:
- Phase 1: Contract creation
- Phase 2: Root solution generation
- Phase 3: Evolutionary optimization loop
"""

import argparse
import os
import sys
import subprocess
import json
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from constants import (
    RESULTS_FILE, PROPOSAL_POOL_DIR, MAX_EVOLUTIONARY_ITERATIONS,
    MUTATION_BATCH, SELECTION_POOL_SIZE, USER_INPUT_DIR, AB_DIR,
    MAX_CHILDREN_PER_NODE, DATASET_CONFIG_PATH
)
from retrieve_champion import retrieve_champion
from agents import get_available_gpus
from select_mutations import select_solutions_for_mutation
from propose_critic import generate_proposal
from retrieve_KB import load_problem_description


def load_results_json():
    """Load results.json file"""
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return {}


def run_data_analysis_phase():
    """
    Phase 0.5: Run data analysis on training dataset (if provided)

    This phase runs BEFORE contract creation to analyze training data characteristics.
    """
    print("\n" + "="*80)
    print("PHASE 0.5: DATA ANALYSIS")
    print("="*80)

    # Check if dataset config exists
    if not os.path.exists(DATASET_CONFIG_PATH):
        print("No dataset_config.json found in USER_INPUT/")
        print("Skipping data analysis phase\n")
        return

    # Load config
    try:
        with open(DATASET_CONFIG_PATH, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in dataset_config.json: {e}")
        raise RuntimeError("Failed to parse dataset_config.json")

    # Check if training set exists
    if "training_set" not in config:
        print("No training_set found in dataset_config.json")
        print("Skipping data analysis phase\n")
        return

    # Verify training file exists
    training_file = config["training_set"].get("filename", "")
    training_path = os.path.join(USER_INPUT_DIR, training_file)

    if not os.path.exists(training_path):
        print(f"Error: Training file not found: {training_path}")
        raise FileNotFoundError(f"Dataset file specified in config does not exist: {training_file}")

    print(f"Training dataset detected: {training_file}")
    print("Starting exploratory data analysis...\n")

    # Run data analysis
    from run_data_analysis import main as data_analysis_main
    result = data_analysis_main()

    if result != 0:
        print("\n⚠ Warning: Data analysis failed")
        print("Continuing without analysis report...")
    else:
        print("\n✓ Phase 0.5 complete: Data analysis report generated")


def run_contract_creation(auto_approve=False):
    """
    Phase 1: Run contract creation with optional auto-approval

    Args:
        auto_approve: If True, automatically approve contract without human review
    """
    print("\n" + "="*80)
    print("PHASE 1: CONTRACT CREATION")
    print("="*80)

    from create_contract import main as contract_main
    result = contract_main(auto_approve=auto_approve)

    if result != 0:
        raise RuntimeError(f"Contract creation failed with return code {result}")

    print("\n✓ Phase 1 complete: Contract created")


def run_root_solution():
    """
    Phase 2: Run root solution generation
    """
    print("\n" + "="*80)
    print("PHASE 2: ROOT SOLUTION GENERATION")
    print("="*80)

    result = subprocess.run(
        [sys.executable, "create_root.py"],
        cwd=os.path.dirname(__file__) or ".",
        capture_output=False
    )

    if result.returncode != 0:
        raise RuntimeError(f"Root solution failed with return code {result.returncode}")

    print("\n✓ Phase 2 complete: Root solution created")


# ============================================================================
# Helper Functions for Batch Mutation
# ============================================================================

def get_top_k_solutions(k: int) -> dict:
    """
    Get top-K solutions from results.json sorted by score.

    Args:
        k: Number of top solutions to retrieve

    Returns:
        Dictionary mapping solution_id to result data for top-K solutions
    """
    if not os.path.exists(RESULTS_FILE):
        return {}

    with open(RESULTS_FILE, 'r') as f:
        results = json.load(f)

    if not results:
        return {}

    # Filter to successful solutions only if we have enough
    successful = {sid: data for sid, data in results.items()
                  if data.get('status') == 'success'}

    if len(successful) >= k:
        results_to_rank = successful
    else:
        # Not enough successful solutions, include all
        results_to_rank = results

    # Sort by score (ascending, lower is better)
    sorted_solutions = sorted(
        results_to_rank.items(),
        key=lambda x: x[1].get('score', float('inf'))
    )

    # Take top K
    top_k = dict(sorted_solutions[:k])

    return top_k


def get_child_count(parent_id: str) -> int:
    """
    Count existing children of a parent solution in results.json.

    Args:
        parent_id: Parent solution ID (e.g., "solution_0")

    Returns:
        Number of existing children (0-9)
    """
    if not os.path.exists(RESULTS_FILE):
        return 0

    with open(RESULTS_FILE, 'r') as f:
        results = json.load(f)

    # Count solutions whose parent_id matches
    count = sum(1 for data in results.values()
                if data.get('parent_id') == parent_id)

    return count


def select_parents_for_mutation(iteration: int) -> list[dict]:
    """
    Select parent solutions for mutation (early or mature stage).

    Early stage (≤ MUTATION_BATCH solutions): All successful solutions mutate
    Mature stage (> MUTATION_BATCH): Ensemble selects MUTATION_BATCH-1, best always included

    Args:
        iteration: Current iteration number

    Returns:
        List of parent selections with 'solution_id' and 'selector_reasoning'
    """
    print("\n" + "="*80)
    print("PARENT SELECTION")
    print("="*80)

    if not os.path.exists(RESULTS_FILE):
        raise RuntimeError("No results.json found - cannot select parents")

    with open(RESULTS_FILE, 'r') as f:
        results = json.load(f)

    # Get successful solutions with room for more children
    successful = [sid for sid, data in results.items()
                  if data.get('status') == 'success'
                  and get_child_count(sid) < MAX_CHILDREN_PER_NODE]

    print(f"Total solutions: {len(results)}")
    print(f"Successful solutions: {len(successful)}")

    # Early stage: all successful solutions mutate
    if len(successful) <= MUTATION_BATCH:
        print(f"\n[EARLY STAGE] All {len(successful)} solutions will mutate")

        selections = []
        for solution_id in successful:
            selections.append({
                'solution_id': solution_id,
                'selector_reasoning': "Early stage - all successful solutions mutate"
            })

        return selections

    # Mature stage: ensemble selection
    print(f"\n[MATURE STAGE] Ensemble selects {MUTATION_BATCH-1} + best-scoring")

    # Get best solution that has room for more children
    successful_results = {sid: data for sid, data in results.items()
                         if data.get('status') == 'success'}

    sorted_successful = sorted(successful_results.items(),
                              key=lambda x: x[1].get('score', float('inf')))

    best_solution_id = None
    for sol_id, sol_data in sorted_successful:
        if get_child_count(sol_id) < MAX_CHILDREN_PER_NODE:
            best_solution_id = sol_id
            break

    if best_solution_id is None:
        print("\n⚠ WARNING: All successful solutions have reached MAX_CHILDREN_PER_NODE")
        print("  No parents available for mutation - skipping iteration")
        return []

    print(f"\nBest-scoring solution (with room for children): {best_solution_id}")

    # Get top-(K+1) solutions with room for children, then exclude best for ensemble review
    # This ensures ensemble sees ranks 2 through K+1, never the best
    top_k_plus_one = get_top_k_solutions(SELECTION_POOL_SIZE + 1)
    ensemble_pool = {sid: data for sid, data in top_k_plus_one.items()
                     if sid != best_solution_id
                     and get_child_count(sid) < MAX_CHILDREN_PER_NODE}

    print(f"\nEnsemble pool ({len(ensemble_pool)} solutions, excluding best):")
    for sid, data in list(ensemble_pool.items())[:5]:
        print(f"  {sid}: score={data.get('score', 'N/A')}")

    # Check if ensemble pool is sufficient
    if len(ensemble_pool) < MUTATION_BATCH - 1:
        print(f"\n⚠ WARNING: Ensemble pool only has {len(ensemble_pool)} solutions")
        print(f"  Need {MUTATION_BATCH - 1} for full ensemble selection")
        print(f"  Proceeding with available solutions...")

    # Load problem description
    problem = load_problem_description()
    requirements = problem  # Requirements included in problem description

    # Run ensemble selection (ensemble never sees best solution)
    print(f"\nRunning ensemble selection...")
    selection_result = select_solutions_for_mutation(
        problem=problem,
        requirements=requirements,
        top_k_results=ensemble_pool,  # Best excluded
        iteration=iteration
    )

    # Extract ensemble selections (already MUTATION_BATCH-1)
    ensemble_selections = selection_result['selected']

    print(f"\nEnsemble selected {len(ensemble_selections)} solutions:")
    for sel in ensemble_selections:
        print(f"  {sel['solution_id']}: {sel['vote_count']} votes")

    # Build final selections: best first, then ensemble picks
    selections = []

    # Add best-scoring first
    selections.append({
        'solution_id': best_solution_id,
        'selector_reasoning': "Best-scoring solution (always selected for mutation)"
    })

    # Add all ensemble selections (no deduplication needed - best was never in pool)
    for sel in ensemble_selections:
        selections.append({
            'solution_id': sel['solution_id'],
            'selector_reasoning': sel['aggregated_reasoning']
        })

    print(f"\n✓ Final parent selections: {len(selections)}")
    for sel in selections:
        print(f"  {sel['solution_id']}")

    return selections


def generate_proposals_batch(parent_selections: list[dict], iteration: int) -> list[tuple]:
    """
    Generate proposals for all selected parents in parallel.

    Args:
        parent_selections: List of parent selections from select_parents_for_mutation
        iteration: Current iteration number

    Returns:
        List of (child_id, parent_id) tuples
    """
    print("\n" + "="*80)
    print("PROPOSAL GENERATION (PARALLEL)")
    print("="*80)
    print(f"Generating {len(parent_selections)} proposals...")

    def generate_single_proposal(parent_data: dict) -> tuple:
        """Wrapper for parallel proposal generation"""
        parent_id = parent_data['solution_id']
        selector_reasoning = parent_data['selector_reasoning']

        # Generate child ID based on existing children count
        child_index = get_child_count(parent_id)
        parent_numeric = parent_id.replace('solution_', '')
        child_id = f"solution_{parent_numeric}{child_index}"

        print(f"\n[{parent_id}] Generating proposal for {child_id}...")

        try:
            proposal, discussion = generate_proposal(
                parent_id=parent_id,
                child_id=child_id,
                selector_reasoning=selector_reasoning
            )
            print(f"[{parent_id}] ✓ Proposal generated: {child_id}")
            return (child_id, parent_id)
        except Exception as e:
            print(f"[{parent_id}] ✗ ERROR: {e}")
            return None

    # Run proposals in parallel
    child_parent_pairs = []
    with ThreadPoolExecutor(max_workers=MUTATION_BATCH) as executor:
        futures = {executor.submit(generate_single_proposal, parent): parent
                   for parent in parent_selections}

        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                child_parent_pairs.append(result)

    print(f"\n✓ Generated {len(child_parent_pairs)} proposals")
    return child_parent_pairs


def engineer_children_batch(child_parent_pairs: list[tuple], gpu_ids: list[int]) -> dict:
    """
    Engineer all children in parallel with dynamic GPU distribution.

    Args:
        child_parent_pairs: List of (child_id, parent_id) tuples
        gpu_ids: List of available GPU IDs (empty for CPU only)

    Returns:
        Dictionary mapping child_id to result dict
    """
    print("\n" + "="*80)
    print("ENGINEERING CHILDREN (PARALLEL)")
    print("="*80)
    print(f"Engineering {len(child_parent_pairs)} children...")
    print(f"Available GPUs: {gpu_ids if gpu_ids else 'CPU only'}")

    # Assign GPUs round-robin
    child_gpu_assignments = []
    for i, (child_id, parent_id) in enumerate(child_parent_pairs):
        if gpu_ids:
            gpu = gpu_ids[i % len(gpu_ids)]
        else:
            gpu = None
        child_gpu_assignments.append((child_id, gpu))
        print(f"  {child_id} → GPU {gpu if gpu is not None else 'CPU'}")

    # Run engineering in parallel
    results = {}
    with ProcessPoolExecutor(max_workers=len(child_parent_pairs)) as executor:
        futures = {}
        for child_id, gpu in child_gpu_assignments:
            future = executor.submit(engineer_child_wrapper, child_id, gpu)
            futures[future] = child_id

        for future in as_completed(futures):
            child_id = futures[future]
            try:
                result = future.result()
                results[child_id] = result
                status = result.get('status', 'unknown')
                print(f"\n[{child_id}] Completed with status: {status}")
            except Exception as e:
                print(f"\n[{child_id}] ERROR: {e}")
                results[child_id] = {
                    'solution_id': child_id,
                    'status': 'error',
                    'error': str(e)
                }

    print(f"\n✓ All children engineered")
    return results


def analyze_children_batch(child_parent_pairs: list[tuple]):
    """
    Analyze all children in parallel.

    Args:
        child_parent_pairs: List of (child_id, parent_id) tuples
    """
    print("\n" + "="*80)
    print("ANALYZING CHILDREN (PARALLEL)")
    print("="*80)
    print(f"Analyzing {len(child_parent_pairs)} children...")

    def analyze_single_child(child_id: str, parent_id: str):
        """Wrapper for parallel analysis"""
        try:
            print(f"\n[{child_id}] Analyzing...")

            numeric_id = child_id.replace('solution_', '')
            proposal_file = os.path.join(PROPOSAL_POOL_DIR, f"proposal_{numeric_id}.md")

            _env = os.environ.copy()
            _env['SCIML_TELEMETRY_SOLUTION_ID'] = child_id
            result = subprocess.run(
                [
                    sys.executable, "analyze.py",
                    "--solution_id", child_id,
                    "--parent_id", parent_id,
                    "--proposal_file", proposal_file
                ],
                cwd=os.path.dirname(__file__) or ".",
                capture_output=True,
                text=True,
                env=_env,
            )

            if result.returncode != 0:
                print(f"[{child_id}] ✗ Analysis failed: {result.stderr}")
                return False

            print(f"[{child_id}] ✓ Analysis complete")
            return True

        except Exception as e:
            print(f"[{child_id}] ✗ ERROR: {e}")
            return False

    # Run analysis in parallel
    success_count = 0
    with ThreadPoolExecutor(max_workers=len(child_parent_pairs)) as executor:
        futures = {executor.submit(analyze_single_child, child_id, parent_id): child_id
                   for child_id, parent_id in child_parent_pairs}

        for future in as_completed(futures):
            if future.result():
                success_count += 1

    print(f"\n✓ Analysis complete: {success_count}/{len(child_parent_pairs)} succeeded")


def engineer_child_wrapper(solution_id: str, gpu_id: int | None = None) -> dict:
    """
    Wrapper function for parallel execution of engineer_execute_evaluate.

    This runs in a separate process with CUDA_VISIBLE_DEVICES set.

    Args:
        solution_id: Child solution ID
        gpu_id: GPU ID to assign (None for CPU)

    Returns:
        Dictionary with results from engineer_execute_evaluate
    """
    import os
    import subprocess
    import sys

    # Set CUDA_VISIBLE_DEVICES
    env = os.environ.copy()
    if gpu_id is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        print(f"\n[{solution_id}] Assigned to GPU {gpu_id}")
    else:
        env['CUDA_VISIBLE_DEVICES'] = ''
        print(f"\n[{solution_id}] Running on CPU")

    # Pass solution_id to subprocess for telemetry
    env['SCIML_TELEMETRY_SOLUTION_ID'] = solution_id

    # Run engineer_execute_evaluate.py
    result = subprocess.run(
        [sys.executable, "engineer_execute_evaluate.py", "--solution_id", solution_id],
        cwd=os.path.dirname(__file__) or ".",
        env=env,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"\n[{solution_id}] ERROR during engineering:")
        print(result.stdout)
        print(result.stderr)
        return {
            "solution_id": solution_id,
            "status": "error",
            "error": result.stderr
        }

    # Parse output (engineer_execute_evaluate.py prints final results as JSON)
    # For now, just return success
    print(f"\n[{solution_id}] ✓ Engineering complete")

    return {
        "solution_id": solution_id,
        "status": "success"
    }


def run_evolutionary_loop(gpu_ids: list[int]):
    """
    Phase 3: Run evolutionary optimization loop with ensemble-guided batch mutation.

    Args:
        gpu_ids: List of available GPU IDs (auto-detected or user-provided)
    """
    import time as _time
    print("\n" + "="*80)
    print("PHASE 3: EVOLUTIONARY OPTIMIZATION")
    print("="*80)
    print(f"Max iterations: {MAX_EVOLUTIONARY_ITERATIONS}")
    print(f"Mutation batch size: {MUTATION_BATCH}")
    print(f"GPU IDs: {gpu_ids if gpu_ids else 'CPU only'}")

    # Set up telemetry directory (RESULTS dir for this run)
    _tel_dir = os.environ.get("SCIML_TELEMETRY_DIR")
    if not _tel_dir:
        # Default to RESULTS dir if not set externally
        from constants import RESULTS_DIR
        _tel_dir = RESULTS_DIR
        os.environ["SCIML_TELEMETRY_DIR"] = _tel_dir
    os.makedirs(_tel_dir, exist_ok=True)
    _loop_start = _time.time()

    for iteration in range(MAX_EVOLUTIONARY_ITERATIONS):
        # Update iteration env var so subprocesses inherit it
        os.environ["SCIML_TELEMETRY_ITERATION"] = str(iteration + 1)
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration + 1}/{MAX_EVOLUTIONARY_ITERATIONS}")
        print(f"{'='*80}")

        # Select parents for mutation (early or mature stage)
        try:
            parent_selections = select_parents_for_mutation(iteration + 1)

            if not parent_selections:
                print("\nNo parents selected - stopping evolutionary loop")
                break

            print(f"\n✓ Selected {len(parent_selections)} parents for mutation")

        except Exception as e:
            print(f"\nERROR selecting parents: {e}")
            import traceback
            traceback.print_exc()
            print("Stopping evolutionary loop.")
            break

        # Generate proposals in parallel
        try:
            child_parent_pairs = generate_proposals_batch(parent_selections, iteration + 1)

            if not child_parent_pairs:
                print("\nNo proposals generated - stopping evolutionary loop")
                break

            print(f"\n✓ Generated {len(child_parent_pairs)} proposals")

        except Exception as e:
            print(f"\nERROR generating proposals: {e}")
            import traceback
            traceback.print_exc()
            print("Stopping evolutionary loop.")
            break

        # Engineer children in parallel
        try:
            results = engineer_children_batch(child_parent_pairs, gpu_ids)

            # Check how many succeeded
            success_count = sum(1 for r in results.values() if r.get('status') == 'success')
            print(f"\n✓ Engineering complete: {success_count}/{len(results)} succeeded")

            if success_count == 0:
                print("\nWARNING: All children failed during engineering")
                # Continue anyway - next iteration will handle this

        except Exception as e:
            print(f"\nERROR during parallel engineering: {e}")
            import traceback
            traceback.print_exc()
            print("Stopping evolutionary loop.")
            break

        # Analyze children in parallel
        try:
            analyze_children_batch(child_parent_pairs)
        except Exception as e:
            print(f"\nWARNING: Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            # Continue anyway - analysis is not critical

        # Print iteration summary
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration + 1} SUMMARY")
        print(f"{'='*80}")
        print(f"Parents mutated: {len(parent_selections)}")
        print(f"Children created: {len(child_parent_pairs)}")
        print(f"New solutions:")
        for child_id, parent_id in child_parent_pairs:
            status = results.get(child_id, {}).get('status', 'unknown')
            print(f"  {child_id} (from {parent_id}): {status}")

    print(f"\n{'='*80}")
    print("EVOLUTIONARY OPTIMIZATION COMPLETE")
    print(f"{'='*80}")

    # Merge telemetry and write summary
    if _tel_dir:
        _loop_duration = _time.time() - _loop_start
        print(f"\n[telemetry] Total evolution wall-clock time: {_loop_duration:.1f}s ({_loop_duration/3600:.2f}h)")
        from telemetry import merge_telemetry
        from constants import RESULTS_DIR
        _summary_path = os.path.join(RESULTS_DIR, "telemetry_summary.json")
        merge_telemetry(_tel_dir, _summary_path)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="SciML Agent Evolutionary System Orchestrator"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "contract-only", "root-only", "evolve-only"],
        help="Execution mode (default: full)"
    )

    parser.add_argument(
        "--gpu_ids",
        type=int,
        nargs='*',
        default=None,
        help=(
            "Available GPU IDs (e.g., --gpu_ids 0 1). "
            "Use --gpu_ids with no values for CPU-only. "
            "If not specified, GPUs are auto-detected."
        )
    )

    args = parser.parse_args()

    # Auto-detect GPUs if not manually specified
    if args.gpu_ids is None:
        detected_gpus = get_available_gpus()
        gpu_ids = detected_gpus
        gpu_source = "auto-detected"
    elif len(args.gpu_ids) == 0:
        gpu_ids = []
        gpu_source = "user-forced-cpu"
    else:
        gpu_ids = args.gpu_ids
        gpu_source = "user-specified"

    print("\n" + "="*80)
    print("SCIML AGENT EVOLUTIONARY SYSTEM")
    print("="*80)
    print(f"Mode: {args.mode}")
    print(f"Max evolutionary iterations: {MAX_EVOLUTIONARY_ITERATIONS}")
    print(f"Mutation batch size: {MUTATION_BATCH}")
    print(f"GPU IDs ({gpu_source}): {gpu_ids if gpu_ids else 'CPU only'}")
    print("="*80)

    try:
        # Phase 0.5: Data Analysis (runs before contract if training data exists)
        if args.mode in ["full", "contract-only"]:
            run_data_analysis_phase()

        # Phase 1: Contract creation
        if args.mode in ["full", "contract-only"]:
            # Auto-approve in full mode for non-interactive batch runs
            auto_approve = (args.mode == "full")
            run_contract_creation(auto_approve=auto_approve)

        # Phase 2: Root solution
        if args.mode in ["full", "root-only"]:
            run_root_solution()

        # Phase 3: Evolutionary loop
        if args.mode in ["full", "evolve-only"]:
            run_evolutionary_loop(gpu_ids)

        print("\n" + "="*80)
        print("ALL PHASES COMPLETE")
        print("="*80)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting gracefully...")
        sys.exit(1)

    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
