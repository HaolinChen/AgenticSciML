"""
Selection agent for ensemble-guided batch mutation.

Uses Triple-G ensemble (GPT, Grok, Gemini) to vote on which solutions
have the most potential for further mutation.
"""

import os
import json
import time
from typing import List, Dict
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from constants import (
    MUTATION_BATCH, SELECTION_POOL_SIZE, ENSEMBLE_MODELS,
    SELECTION_LOGS_DIR, MODELS
)


# ============================================================================
# Pydantic Models
# ============================================================================

class SolutionSelection(BaseModel):
    """Single solution selection with reasoning"""
    solution_id: str = Field(description="Solution ID (e.g., 'solution_012')")
    reasoning: str = Field(description="2-3 bullet points explaining why this solution has potential")


class SelectionOutput(BaseModel):
    """Structured output from selector agent"""
    selections: List[SolutionSelection] = Field(
        description="Selected solutions for mutation (count based on current MUTATION_BATCH setting)"
    )


# ============================================================================
# Helper Functions
# ============================================================================

def get_selector_llm(model_name: str):
    """Get LLM instance for selector agent"""
    temperature = 0.5
    max_tokens = 8192

    if "claude" in model_name:
        return ChatAnthropic(model=model_name, temperature=temperature, max_tokens=max_tokens)
    elif "gpt" in model_name:
        return ChatOpenAI(model=model_name, temperature=temperature, max_tokens=max_tokens)
    elif "gemini" in model_name:
        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature, max_output_tokens=max_tokens)
    elif "grok" in model_name:
        # Grok uses OpenAI-compatible API
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url="https://api.x.ai/v1",
            api_key=os.getenv("XAI_API_KEY")
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def format_solution_context(solution_id: str, solution_data: dict,
                            solution_code: str, analysis: str) -> str:
    """Format a single solution's context for the prompt"""
    parent_id = solution_data.get("parent_id", "None")
    score = solution_data.get("score", float('inf'))
    status = solution_data.get("status", "unknown")

    return f"""**{solution_id}** (score: {score:.6f}, parent: {parent_id}, status: {status})

Code:
```python
{solution_code}
```

Analysis:
{analysis}

---"""


# ============================================================================
# Selector Prompt
# ============================================================================

SELECTOR_SYSTEM_PROMPT = """You are a solution selector for scientific machine learning research. Your task is to identify solutions with the most potential for further improvement through mutation.

Your role is CRITICAL: You must balance exploitation (refining promising solutions) and exploration (trying underexplored branches). Consider both performance metrics AND structural insights from code and analysis.

**Key Responsibilities:**
1. Review top solutions' code, analysis reports, and scores
2. Identify which solutions show the most promise for mutation
3. Consider both high-performers (refinement potential) and interesting failures (fixable issues)
4. Balance exploitation vs exploration across the selection pool

**Selection Criteria - Indicators of Potential:**

✅ **HIGH POTENTIAL (select these):**
- New architecture with moderate losses → may need hyperparameter tuning only
- Innovative techniques with implementation flaws → bugs are fixable
- Strong theoretical foundation but imbalanced loss terms → needs rebalancing
- Novel approach that hasn't converged yet → more training or tweaks needed
- Underexplored branch (fewer children) → unexplored search space
- Good ideas with suboptimal hyperparameters → easy wins available

❌ **LOW POTENTIAL (avoid these):**
- Traditional architecture with performance plateau → diminishing returns
- Repeatedly failed mutations with no clear improvement path
- Fundamental design flaws incompatible with problem requirements
- Overexplored branches (many children already) → likely exhausted

**CRITICAL - Exploitation vs Exploration Trade-off:**
- Exploitation: Select 1-2 top performers that can be incrementally improved
- Exploration: Select 1-2 interesting approaches that failed but have fixable issues or unexplored potential
- DO NOT select only the best-scoring solutions (that's pure exploitation)
- DO NOT select only novel/risky solutions (that's pure exploration)
- Balance is key for effective search"""


def generate_selector_prompt(problem_description: str, requirements: str,
                             top_k_solutions: List[tuple], iteration: int) -> str:
    """
    Generate prompt for selector agent.

    Args:
        problem_description: User's problem description
        requirements: User's requirements
        top_k_solutions: List of (solution_id, solution_data, code_snippet, analysis_excerpt) tuples
        iteration: Current iteration number

    Returns:
        Formatted prompt string
    """

    # Format solutions for prompt
    solutions_text = "\n\n".join([
        format_solution_context(sol_id, sol_data, code, analysis)
        for sol_id, sol_data, code, analysis in top_k_solutions
    ])

    prompt = f"""You are selecting {MUTATION_BATCH-1} solutions from the top {SELECTION_POOL_SIZE} for mutation in iteration {iteration}.

**PROBLEM CONTEXT:**
{problem_description}

**REQUIREMENTS:**
{requirements}

**SELECTION TASK:**
Review the top {len(top_k_solutions)} solutions below and select exactly {MUTATION_BATCH-1} solutions that show the most promise for mutation.

Consider:
1. **Exploitation vs Exploration trade-off** - select a mix of high-performers and interesting failures
2. **Indicators of potential** - new architectures, innovative ideas, fixable bugs, unexplored branches
3. **Red flags** - traditional approaches with plateau, repeated failures, fundamental flaws

**TOP {len(top_k_solutions)} CANDIDATE SOLUTIONS:**

{solutions_text}

**FEW-SHOT EXAMPLES:**

EXAMPLE 1 (HIGH POTENTIAL - Select):
solution_012 (score: 0.045):
- New physics-informed architecture with residual connections
- Training log shows steady loss decrease, not yet converged
- Analysis: "Boundary condition handling is weak (BC error: 0.08), but PDE residuals are excellent (0.002)"
REASONING: ✓ Architecture is sound, just needs loss rebalancing between BC and PDE terms

EXAMPLE 2 (LOW POTENTIAL - Skip):
solution_034 (score: 0.092):
- Standard MLP with ReLU activations
- Training log shows early plateau after epoch 50
- Analysis: "Simple architecture, exhausted learning capacity, tried 5 children already"
REASONING: ✗ Traditional approach with no clear improvement path, already heavily explored

EXAMPLE 3 (HIGH POTENTIAL - Select):
solution_05 (score: 0.067):
- Novel adaptive weighting scheme for multi-objective loss
- Training log shows oscillating losses
- Analysis: "Innovative idea, but weights change too aggressively (oscillations every 10 epochs)"
REASONING: ✓ Innovative approach, needs hyperparameter adjustment (damping factor), high exploration value

EXAMPLE 4 (LOW POTENTIAL - Skip):
solution_08 (score: 0.053):
- Standard DeepONet architecture
- Training log shows smooth convergence
- Analysis: "Well-optimized implementation, performance near theoretical limit"
REASONING: ✗ Already well-optimized, limited room for improvement

**YOUR TASK:**
Select exactly {MUTATION_BATCH-1} solutions and provide concise reasoning (2-3 bullet points) for each.

Output format:
- solution_XX: [Why it has potential, what improvements are expected]
- solution_YY: [Why it has potential, what improvements are expected]
- solution_ZZ: [Why it has potential, what improvements are expected]"""

    return prompt


# ============================================================================
# Ensemble Selection Logic
# ============================================================================

def run_single_selector(model_key: str, model_name: str, problem: str, requirements: str,
                       top_k_solutions: List[tuple], iteration: int) -> tuple:
    """
    Run a single selector model (wrapper for parallel execution).

    Args:
        model_key: Model key (e.g., "gpt", "grok", "gemini")
        model_name: Model name (e.g., "gpt-4o", "gemini-2.5-pro")
        problem: Problem description
        requirements: User requirements
        top_k_solutions: List of solution tuples
        iteration: Current iteration

    Returns:
        Tuple of (model_key, SelectionOutput) or (model_key, Exception) if failed
    """
    try:
        llm = get_selector_llm(model_name)
        prompt = generate_selector_prompt(problem, requirements, top_k_solutions, iteration)

        messages = [
            HumanMessage(content=SELECTOR_SYSTEM_PROMPT + "\n\n" + prompt)
        ]

        result = llm.with_structured_output(SelectionOutput).invoke(messages)
        return (model_key, result)
    except Exception as e:
        return (model_key, e)


def aggregate_selections(selections_by_model: Dict[str, SelectionOutput],
                        scores_dict: Dict[str, float]) -> List[Dict]:
    """
    Aggregate selections from multiple models using majority voting.

    Args:
        selections_by_model: Dictionary mapping model name to SelectionOutput
        scores_dict: Dictionary mapping solution_id to score (for tiebreaking)

    Returns:
        List of selected solutions with aggregated reasoning
    """
    # Count votes for each solution
    vote_counter = Counter()
    reasoning_by_solution = {}

    for model_name, selection_output in selections_by_model.items():
        for selection in selection_output.selections:
            sol_id = selection.solution_id
            vote_counter[sol_id] += 1

            # Collect reasoning from all models that voted for this solution
            if sol_id not in reasoning_by_solution:
                reasoning_by_solution[sol_id] = []
            reasoning_by_solution[sol_id].append({
                "model": model_name,
                "reasoning": selection.reasoning
            })

    # Get top (mutation_batch-1) solutions by vote count
    # Tiebreaker: lower score is better
    ranked_solutions = sorted(
        vote_counter.items(),
        key=lambda x: (-x[1], scores_dict.get(x[0], float('inf')))  # Sort by votes (desc), then score (asc)
    )

    # Select top (mutation_batch-1)
    selected = ranked_solutions[:MUTATION_BATCH-1]

    # Build final selection list with aggregated reasoning
    final_selections = []
    for sol_id, vote_count in selected:
        # Aggregate reasoning from all models
        all_reasoning = reasoning_by_solution[sol_id]
        aggregated_reasoning = "\n".join([
            f"- [{r['model']}] {r['reasoning']}" for r in all_reasoning
        ])

        final_selections.append({
            "solution_id": sol_id,
            "vote_count": vote_count,
            "aggregated_reasoning": aggregated_reasoning
        })

    return final_selections


# ============================================================================
# Main Selection Function
# ============================================================================

def select_solutions_for_mutation(problem: str, requirements: str,
                                 top_k_results: Dict[str, dict],
                                 iteration: int) -> Dict:
    """
    Run ensemble selection to choose solutions for mutation.

    Uses ThreadPoolExecutor to run all selector models in parallel.

    Args:
        problem: Problem description
        requirements: User requirements
        top_k_results: Dictionary of top-K solution results
        iteration: Current iteration number

    Returns:
        Dictionary with:
        - selected: List of selected solutions with aggregated reasoning
        - voting_results: Detailed voting breakdown
        - log_file: Path to selection log
    """
    from constants import SOLUTION_AND_OUTPUTS_DIR, AB_DIR


    print(f"\n{'='*80}")
    print(f"ENSEMBLE SELECTION - Iteration {iteration}")
    print(f"{'='*80}\n")

    # Load solution codes and analyses (FULL CONTEXT - no truncation)
    top_k_solutions = []
    for sol_id, sol_data in top_k_results.items():
        # Load solution code (FULL)
        solution_path = os.path.join(SOLUTION_AND_OUTPUTS_DIR, sol_id, "solution.py")
        if os.path.exists(solution_path):
            with open(solution_path, 'r') as f:
                solution_code = f.read()
        else:
            solution_code = "(Code not found)"

        # Load analysis (FULL)
        analysis_path = os.path.join(AB_DIR, f"{sol_id}_analysis.md")
        if os.path.exists(analysis_path):
            with open(analysis_path, 'r') as f:
                analysis_md = f.read()
        else:
            analysis_md = "(Analysis not found)"

        top_k_solutions.append((sol_id, sol_data, solution_code, analysis_md))

    # Run each selector model IN PARALLEL using ThreadPoolExecutor
    print(f"Running {len(ENSEMBLE_MODELS)} selector models in parallel...")
    selections_by_model = {}

    with ThreadPoolExecutor(max_workers=len(ENSEMBLE_MODELS)) as executor:
        # Submit all tasks
        futures = {}
        for model_key in ENSEMBLE_MODELS:
            model_name = MODELS[model_key]
            future = executor.submit(
                run_single_selector,
                model_key, model_name, problem, requirements, top_k_solutions, iteration
            )
            futures[future] = model_key

        # Collect results as they complete
        for future in as_completed(futures):
            model_key = futures[future]
            result = future.result()

            if isinstance(result[1], Exception):
                print(f"✗ {model_key} failed: {result[1]}")
            else:
                selection_output = result[1]
                selections_by_model[model_key] = selection_output
                selected_ids = [s.solution_id for s in selection_output.selections]
                print(f"✓ {model_key} selected: {selected_ids}")

    if not selections_by_model:
        raise RuntimeError("All selector models failed!")

    # Aggregate votes
    scores_dict = {sol_id: sol_data["score"] for sol_id, sol_data in top_k_results.items()}
    final_selections = aggregate_selections(selections_by_model, scores_dict)

    print(f"\n✓ Final selections (after voting):")
    for sel in final_selections:
        print(f"  - {sel['solution_id']} ({sel['vote_count']} votes)")

    # Track voting results
    votes_per_model = {
        model_key: [s.solution_id for s in output.selections]
        for model_key, output in selections_by_model.items()
    }
    vote_counts = {sel['solution_id']: sel['vote_count'] for sel in final_selections}
    aggregated_result = [sel['solution_id'] for sel in final_selections]


    # Track timing

    # Save selection log
    log_file = save_selection_log(
        iteration, top_k_results,
        selections_by_model, final_selections
    )

    print(f"✓ Selection log saved: {log_file}\n")

    return {
        "selected": final_selections,
        "voting_results": selections_by_model,
        "log_file": log_file
    }


def save_selection_log(iteration: int,
                      top_k_results: Dict[str, dict],
                      selections_by_model: Dict[str, SelectionOutput],
                      final_selections: List[Dict]) -> str:
    """
    Save selection log to SELECTION_LOGS directory.

    Args:
        iteration: Current iteration number
        top_k_results: Top-K solution results
        selections_by_model: Selections from each model
        final_selections: Final aggregated selections

    Returns:
        Path to log file
    """
    os.makedirs(SELECTION_LOGS_DIR, exist_ok=True)
    log_file = os.path.join(SELECTION_LOGS_DIR, f"iteration_{iteration:03d}.md")

    # Build log content
    log_lines = [
        f"# Selection Log - Iteration {iteration}",
        "",
        "## Top Candidate Solutions",
        ""
    ]

    # List candidates
    for sol_id, sol_data in top_k_results.items():
        score = sol_data.get("score", float('inf'))
        parent_id = sol_data.get("parent_id", "None")
        status = sol_data.get("status", "unknown")
        log_lines.append(f"- **{sol_id}**: score={score:.6f}, parent={parent_id}, status={status}")

    log_lines.extend(["", "---", ""])

    # Individual model selections
    log_lines.append("## Individual Model Selections")
    log_lines.append("")

    for model_key, selection_output in selections_by_model.items():
        log_lines.append(f"### {model_key.upper()}")
        log_lines.append("")
        for i, selection in enumerate(selection_output.selections, 1):
            log_lines.append(f"{i}. **{selection.solution_id}**")
            log_lines.append(f"   {selection.reasoning}")
            log_lines.append("")

    log_lines.extend(["---", ""])

    # Voting results
    log_lines.append("## Voting Results")
    log_lines.append("")

    # Count votes for all solutions
    all_votes = Counter()
    for selection_output in selections_by_model.values():
        for selection in selection_output.selections:
            all_votes[selection.solution_id] += 1

    for sol_id, count in all_votes.most_common():
        voters = [
            model_key for model_key, sel_out in selections_by_model.items()
            if sol_id in [s.solution_id for s in sel_out.selections]
        ]
        log_lines.append(f"- **{sol_id}**: {count} votes ({', '.join(voters)})")

    log_lines.extend(["", "---", ""])

    # Final selection
    log_lines.append("## Final Selection")
    log_lines.append("")
    log_lines.append(f"Selected {len(final_selections)} solutions for mutation:")
    log_lines.append("")

    for i, sel in enumerate(final_selections, 1):
        log_lines.append(f"### {i}. {sel['solution_id']} ({sel['vote_count']} votes)")
        log_lines.append("")
        log_lines.append("**Aggregated Reasoning:**")
        log_lines.append(sel['aggregated_reasoning'])
        log_lines.append("")

    # Write to file
    with open(log_file, 'w') as f:
        f.write('\n'.join(log_lines))

    return log_file


# ============================================================================
# Main (for testing)
# ============================================================================

def main():
    """Test the selector with mock data"""
    # This would be used for unit testing
    print("Use test_select_mutations.py for testing")


if __name__ == "__main__":
    main()
