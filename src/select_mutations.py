"""
Selection agent for ensemble-guided batch mutation.

Uses Triple-G ensemble (GPT, Grok, Gemini) to vote on which solutions
have the most potential for further mutation.

================================================================================
中文模块说明（学习注释）
================================================================================
作用：
    本模块是 AgenticSciML **Phase3 进化循环** 的“集成选择器(selector)”。
    在每次迭代中，用多个大模型组成的集成(ENSEMBLE_MODELS，如 gpt/grok/gemini)
    对 Top-K 候选解各自“投票”挑出最有变异潜力的解，再通过多数投票聚合出本轮
    要变异的亲本集合，并附带选择理由。强调“开发(exploitation) vs 探索(exploration)”平衡。

所属阶段：
    Phase3（进化循环）：**选亲本(本模块)** → 多智能体辩论提案(propose_critic)
    → 工程化子代 → 分析(analyze)。

主要输入：
    - 参数：problem（问题描述）、requirements（需求）、top_k_results（Top-K 解结果字典，
        含 score/parent_id/status 等）、iteration（当前迭代号）。
    - 文件：
        * {SOLUTION_AND_OUTPUTS_DIR}/{sol_id}/solution.py —— 各候选解源码（完整）
        * {AB_DIR}/{sol_id}_analysis.md                   —— 各候选解分析报告（完整）
    - 环境变量：各家模型 API Key（如 grok 用 XAI_API_KEY；gpt/gemini/claude 依赖各自默认 Key）。
    - 配置常量：MUTATION_BATCH（本轮变异总数，选择数=其减 1）、SELECTION_POOL_SIZE、
        ENSEMBLE_MODELS、MODELS、SELECTION_LOGS_DIR。

主要输出：
    - 文件：{SELECTION_LOGS_DIR}/iteration_XXX.md —— 本轮选择日志（候选、各模型投票、
        投票统计、最终选择与聚合理由）。
    - 返回：select_solutions_for_mutation 返回 dict：
        {"selected": 最终选择列表, "voting_results": 各模型原始输出, "log_file": 日志路径}。

关键函数清单：
    - SolutionSelection / SelectionOutput：selector 的 pydantic 结构化输出 schema。
    - get_selector_llm：按模型名路由到对应 LLM 客户端。
    - format_solution_context / generate_selector_prompt：拼接单解上下文与完整选择 prompt。
    - run_single_selector：单个模型的一次选择（供并行调用）。
    - aggregate_selections：多模型投票聚合（多数票，分数低者优先破平）。
    - select_solutions_for_mutation：主入口，并行跑集成、聚合、落日志。
    - save_selection_log：把选择过程写成 markdown 日志。
================================================================================
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
    """Single solution selection with reasoning

    中文：单个被选中解的结构化条目，含解 ID 与 2-3 条选择理由。
    """
    solution_id: str = Field(description="Solution ID (e.g., 'solution_012')")
    reasoning: str = Field(description="2-3 bullet points explaining why this solution has potential")


class SelectionOutput(BaseModel):
    """Structured output from selector agent

    中文：单个 selector 模型一次调用的结构化输出，即一组 SolutionSelection。
    LLM 通过 with_structured_output 强制按此 schema 返回。
    """
    selections: List[SolutionSelection] = Field(
        description="Selected solutions for mutation (count based on current MUTATION_BATCH setting)"
    )


# ============================================================================
# Helper Functions
# ============================================================================

def get_selector_llm(model_name: str):
    """Get LLM instance for selector agent

    中文：根据模型名字符串路由到对应的 LangChain LLM 客户端
    （claude→Anthropic，gpt→OpenAI，gemini→Google，grok→OpenAI 兼容接口+XAI_API_KEY）。
    统一 temperature=0.5、max_tokens=8192。未知模型抛 ValueError。
    """
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
    """Format a single solution's context for the prompt

    中文：把单个候选解格式化为 prompt 片段（含分数、亲本、状态、完整代码与完整分析报告），
    供 selector 判断其变异潜力。
    """
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

# 中文：selector 的系统级 prompt（提示词字面量，切勿改动）。
# 核心要求：在“开发(精修高分解) vs 探索(尝试有潜力的失败解/冷门分支)”之间取得平衡，
# 并给出高/低潜力的判定信号，避免只选最高分或只选新奇解。
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

    中文：拼装 selector 的用户级 prompt（含问题背景、需求、全部候选解上下文、
    few-shot 高/低潜力示例、以及“选出恰好 MUTATION_BATCH-1 个解”的任务说明与输出格式）。
    注意：其中的字符串字面量为提示词，切勿改动。
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

    中文：单个 selector 模型的一次完整选择（作为并行任务的包装）。
    组合 system+user prompt 后以结构化输出调用 LLM。
    容错设计：捕获异常并作为 (model_key, Exception) 返回，使某个模型失败不影响其它模型。
    调用 LLM：由 model_name 决定（集成中的一员）。
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

    中文：集成投票聚合的核心。把各模型选出的解累计票数，并收集每个解来自各模型的理由；
    再按“票数降序、分数升序(低分更优，作破平)”排序，取前 MUTATION_BATCH-1 个作为最终选择，
    每个附上跨模型聚合后的理由。返回最终选择列表。
    """
    # Count votes for each solution
    # 中文：遍历每个模型的输出，为其选中的每个解 +1 票，并把该模型给出的理由归档到该解名下。
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
    # 中文：排序键 (-票数, 分数)：票数多者靠前；票数相同则分数低(更优)者靠前。
    ranked_solutions = sorted(
        vote_counter.items(),
        key=lambda x: (-x[1], scores_dict.get(x[0], float('inf')))  # Sort by votes (desc), then score (asc)
    )

    # Select top (mutation_batch-1)
    # 中文：取排名前 MUTATION_BATCH-1 个作为本轮要变异的亲本（少 1 是给“当前最优解”留位）。
    selected = ranked_solutions[:MUTATION_BATCH-1]

    # Build final selection list with aggregated reasoning
    final_selections = []
    for sol_id, vote_count in selected:
        # Aggregate reasoning from all models
        # 中文：把投给该解的所有模型理由合并成一段（标注来源模型），作为聚合理由。
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

    中文：本模块主入口。加载 Top-K 各解的完整代码与分析报告 → 用 ThreadPoolExecutor
    并行运行集成中每个 selector 模型 → 汇总各模型输出 → 投票聚合出最终选择 → 落选择日志。
    容错：单模型失败会被跳过；若全部失败则抛 RuntimeError。
    副作用：读取解目录/分析文件；并行调用多个 LLM；写 SELECTION_LOGS/iteration_XXX.md。
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
    # 中文：集成的每个模型互不依赖，用线程池并行提交，谁先完成先收集，缩短总耗时。
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

            # 中文：run_single_selector 把失败封装为 (model_key, Exception)，此处据此过滤。
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
    # 中文：构造 {解ID: 分数} 供投票破平使用，然后做多数投票聚合得到最终选择。
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

    中文：把本轮选择过程写成 markdown 日志（候选清单 → 各模型选择 → 投票统计 → 最终选择与
    聚合理由），落到 {SELECTION_LOGS_DIR}/iteration_XXX.md，供审计与后续检索记忆使用。
    副作用：新建目录并覆盖写日志文件。返回日志路径。
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
    """Test the selector with mock data

    中文：占位入口，真正的测试见 test_select_mutations.py。
    """
    # This would be used for unit testing
    print("Use test_select_mutations.py for testing")


if __name__ == "__main__":
    main()
