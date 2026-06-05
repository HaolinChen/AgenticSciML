"""
Data Analyst Agent module for exploratory data analysis.

This standalone module performs EDA on training datasets using a Gemini agent
with image understanding capabilities. It generates analysis code, executes it,
analyzes visualizations, and produces text-only reports for downstream agents.
"""

import os
import sys
import json
import time
import subprocess
import base64
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage, MessageLikeRepresentation
from langchain_google_genai import ChatGoogleGenerativeAI

from constants import (
    AGENT_MODELS,
    TEMPERATURES,
    DATA_ANALYSIS_DIR,
    USER_INPUT_DIR,
    DATASET_CONFIG_PATH,
    TIMEOUT_DATA_ANALYSIS
)
from agents import get_llm


# ============================================================================
# Pydantic Models
# ============================================================================

class AnalysisCodeOutput(BaseModel):
    """Structured output for data analyst code generation"""
    analysis_code: str = Field(
        description="Complete Python code for exploratory data analysis"
    )
    analysis_plan: str = Field(
        description="Brief description of what this code will analyze and why"
    )


class AnalysisReportOutput(BaseModel):
    """Structured output for data analyst report generation"""
    report_markdown: str = Field(
        description="Text-only analysis report describing findings from visualizations"
    )


# ============================================================================
# System Prompts
# ============================================================================

ANALYST_CODE_GENERATION_PROMPT = """You are an expert data analyst specializing in scientific machine learning datasets.

**Your Task:** Generate Python code to perform exploratory data analysis on the training dataset.

**Focus Areas:**
1. **Mathematical Properties**:
   - Singularities, discontinuities, irregularities
   - Boundary behavior and edge effects
   - Symmetries or asymmetries
   - Multi-scale phenomena

2. **Data Quality**:
   - Outliers and anomalies
   - Distribution characteristics (skewness, heavy tails)
   - Correlation structures
   - Missing or problematic regions

3. **Solution-Relevant Insights**:
   - Features that impact neural network training
   - Regions requiring special attention (sharp gradients, etc.)
   - Sampling density issues
   - Potential challenges for the solver

**Code Requirements:**
1. Load the dataset using the provided loading instructions
2. Generate informative visualizations (matplotlib)
3. Compute and PRINT numerical statistics (percentiles, ranges, etc.)
4. Save all plots to the specified plots directory (absolute path will be provided)
5. Use try-except blocks to handle potential errors gracefully
6. Print clear section headers for different analyses

**Important:**
- Do NOT use advanced ML libraries (PyTorch, TensorFlow, JAX) for analysis
- Use NumPy, matplotlib, scipy for analysis
- All plots MUST be saved (not just shown)
- Print statistics in text format (downstream agents can't see plots)
- Code must be self-contained and executable
- Use the ABSOLUTE PATH for plots directory that will be provided in the task
"""

ANALYST_REPORT_GENERATION_PROMPT = """You are an expert data analyst writing a report for text-only AI agents.

**Your Task:** Analyze the visualizations from exploratory data analysis and write a concise text-only report.

**Context:** The agents receiving this report CANNOT see images. They need you to describe:
- What the visualizations show
- Key patterns, trends, anomalies
- Mathematical/physical interpretations
- Implications for solution strategies

**Report Structure:**
```markdown
# Data Analysis Report

## Dataset Overview
- Dimensions, shapes, value ranges
- Domain characteristics

## Key Findings

### 1. [Finding Title]
**Observation:** [What you see in the plots]
**Interpretation:** [What it means]
**Implication:** [How it affects solution strategy]

### 2. [Finding Title]
...

## Recommendations for Solution Strategy
- [Specific recommendations based on findings]

## Potential Challenges
- [Challenges the solver might face]
```

**Guidelines:**
- Be concise but specific (1-2 pages max)
- Use concrete numbers (e.g., "90th percentile is 10x larger than median")
- Focus on actionable insights
- Avoid jargon unless necessary
- No references to "the plot shows" - instead write "the distribution is..."
"""


# ============================================================================
# Agent Functions
# ============================================================================

def generate_analysis_code(
    dataset_config: dict,
    problem_description: str,
    error_feedback: str = "",
    plots_dir_absolute: str = None
) -> AnalysisCodeOutput:
    """
    Generate Python code for exploratory data analysis.

    Args:
        dataset_config: Dataset configuration from dataset_config.json
        problem_description: Problem description from USER_INPUT
        error_feedback: Error message from previous execution (for debugging)
        plots_dir_absolute: Absolute path to plots directory

    Returns:
        AnalysisCodeOutput with analysis code and plan
    """
    llm = get_llm("data_analyst")

    # Extract training set info
    training_set = dataset_config.get("training_set", {})
    filename = training_set.get("filename", "")
    description = training_set.get("description", "")
    loading_instructions = training_set.get("loading_instructions", "")

    # Construct prompt
    if error_feedback:
        # Debugging mode
        prompt = f"""The previous analysis code failed with the following error:

```
{error_feedback}
```

Please fix the code and regenerate. Make sure to:
1. Address the specific error mentioned above
2. Add proper error handling
3. Verify file paths are correct

**Original Task Context:**

**Problem:**
{problem_description}

**Dataset Information:**
- Filename: {filename}
- Description: {description}
- Loading Instructions: {loading_instructions}

**Plots Directory (use this ABSOLUTE PATH):**
{plots_dir_absolute}

**Important:** Use the absolute path above for saving plots. Example:
```python
import os
plots_dir = r"{plots_dir_absolute}"
os.makedirs(plots_dir, exist_ok=True)
plt.savefig(os.path.join(plots_dir, 'distribution.png'))
```

Generate corrected Python code for exploratory data analysis.
"""
    else:
        # First generation
        prompt = f"""**Problem:**
{problem_description}

**Training Dataset:**
- Filename: {filename}
- Description: {description}
- Loading Instructions: {loading_instructions}

**Plots Directory (use this ABSOLUTE PATH):**
{plots_dir_absolute}

**Important:** Use the absolute path above for saving plots. Example:
```python
import os
plots_dir = r"{plots_dir_absolute}"
os.makedirs(plots_dir, exist_ok=True)
plt.savefig(os.path.join(plots_dir, 'distribution.png'))
```

Generate Python code to perform exploratory data analysis on this training dataset.

Focus on mathematical properties, data quality, and insights relevant to solving the problem.
"""

    messages = [
        SystemMessage(content=ANALYST_CODE_GENERATION_PROMPT),
        HumanMessage(content=prompt)
    ]

    result = llm.with_structured_output(AnalysisCodeOutput).invoke(messages)

    return result


def execute_analysis_code(analysis_code: str, timeout: int = None) -> tuple[bool, str]:
    """
    Execute analysis code and capture output.

    Args:
        analysis_code: Python code to execute
        timeout: Timeout in seconds (defaults to TIMEOUT_DATA_ANALYSIS)

    Returns:
        Tuple of (success: bool, output/error: str)
    """
    if timeout is None:
        timeout = TIMEOUT_DATA_ANALYSIS

    # Save code to file (use absolute path)
    code_path = os.path.abspath(os.path.join(DATA_ANALYSIS_DIR, "analysis_code.py"))
    os.makedirs(DATA_ANALYSIS_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATA_ANALYSIS_DIR, "plots"), exist_ok=True)

    with open(code_path, 'w') as f:
        f.write(analysis_code)

    # Execute code
    try:
        result = subprocess.run(
            [sys.executable, code_path],
            cwd=USER_INPUT_DIR,  # Execute in USER_INPUT to access datasets
            capture_output=True,
            text=True,
            timeout=timeout
        )

        if result.returncode == 0:
            return True, result.stdout
        else:
            error_msg = f"Exit code {result.returncode}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
            return False, error_msg

    except subprocess.TimeoutExpired:
        return False, f"Execution timeout after {timeout} seconds"
    except Exception as e:
        return False, f"Execution error: {str(e)}"


def analyze_plots_with_vision(
    analysis_code: str,
    execution_output: str,
    dataset_config: dict,
    problem_description: str
) -> AnalysisReportOutput:
    """
    Analyze generated plots using Gemini image understanding.

    Args:
        analysis_code: Executed analysis code
        execution_output: Text output from execution
        dataset_config: Dataset configuration
        problem_description: Problem description

    Returns:
        AnalysisReportOutput with markdown report
    """
    # Get Gemini model with vision capability
    model_name = AGENT_MODELS["data_analyst"]
    temperature = TEMPERATURES["data_analyst"]

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        max_output_tokens=16000
    )

    # Load all plots from DATA_ANALYSIS/plots/
    plots_dir = os.path.join(DATA_ANALYSIS_DIR, "plots")
    plot_files = []
    if os.path.exists(plots_dir):
        plot_files = [
            os.path.join(plots_dir, f)
            for f in os.listdir(plots_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ]
        plot_files.sort()  # Consistent ordering

    if not plot_files:
        # No plots generated, create text-only report
        print("Warning: No plots found, generating report from text output only")

    # Prepare message content with images
    message_parts: list[MessageLikeRepresentation] = []

    # Add text prompt
    training_set = dataset_config.get("training_set", {})
    prompt_text = f"""**Problem:**
{problem_description}

**Training Dataset:**
- Filename: {training_set.get('filename', '')}
- Description: {training_set.get('description', '')}

**Analysis Code Output:**
```
{execution_output}
```

**Analysis Code:**
```python
{analysis_code}
```

**Your Task:**
Analyze the visualizations below and write a concise text-only report for downstream AI agents.

Remember: These agents CANNOT see the images. Describe what you observe in text.
"""
    message_parts.append(prompt_text)

    # Add images
    for plot_path in plot_files:
        try:
            with open(plot_path, 'rb') as f:
                image_bytes = f.read()
                image_b64 = base64.b64encode(image_bytes).decode('utf-8')
                # Use LangChain's image format
                message_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                })
        except Exception as e:
            print(f"Warning: Could not load plot {plot_path}: {e}")

    messages = [
        SystemMessage(content=ANALYST_REPORT_GENERATION_PROMPT),
        HumanMessage(content=message_parts)
    ]

    result = llm.with_structured_output(AnalysisReportOutput).invoke(messages)

    return result


# ============================================================================
# Main Workflow
# ============================================================================

def run_analysis_workflow(max_debug_iterations: int = 3) -> int:
    """
    Main workflow for data analysis with self-debug loop.

    Args:
        max_debug_iterations: Maximum number of debug attempts

    Returns:
        0 on success, non-zero on failure
    """

    print("\n" + "="*80)
    print("DATA ANALYSIS WORKFLOW")
    print("="*80)

    # Load dataset config
    if not os.path.exists(DATASET_CONFIG_PATH):
        print("Error: dataset_config.json not found")
        return 1

    with open(DATASET_CONFIG_PATH, 'r') as f:
        dataset_config = json.load(f)

    if "training_set" not in dataset_config:
        print("Error: No training_set in dataset_config.json")
        return 1

    # Load problem description
    problem_files = ["problem.md", "requirements.md", "evaluation.md"]
    problem_description = ""
    for fname in problem_files:
        fpath = os.path.join(USER_INPUT_DIR, fname)
        if os.path.exists(fpath):
            with open(fpath, 'r') as f:
                problem_description += f"\n\n## {fname}\n\n{f.read()}"

    # Create discussion log
    os.makedirs(DATA_ANALYSIS_DIR, exist_ok=True)
    discussion_path = os.path.join(DATA_ANALYSIS_DIR, "analysis_log.md")
    with open(discussion_path, 'w') as f:
        f.write("# Data Analysis Log\n\n")
        f.write(f"**Dataset:** {dataset_config['training_set']['filename']}\n\n")
        f.write("---\n\n")

    # Compute absolute path for plots directory
    plots_dir_absolute = os.path.abspath(os.path.join(DATA_ANALYSIS_DIR, "plots"))
    os.makedirs(plots_dir_absolute, exist_ok=True)

    # Self-debug loop
    error_feedback = ""
    code_result = None
    output = None

    for iteration in range(1, max_debug_iterations + 1):
        print(f"\n[Iteration {iteration}/{max_debug_iterations}] Generating analysis code...")

        # Generate code
        code_result = generate_analysis_code(
            dataset_config=dataset_config,
            problem_description=problem_description,
            error_feedback=error_feedback,
            plots_dir_absolute=plots_dir_absolute
        )

        # Log timing for code generation

        # Log to discussion
        with open(discussion_path, 'a') as f:
            f.write(f"## Iteration {iteration}\n\n")
            f.write(f"**Plan:** {code_result.analysis_plan}\n\n")
            if error_feedback:
                f.write(f"**Previous Error:**\n```\n{error_feedback}\n```\n\n")
            f.write(f"**Generated Code:**\n```python\n{code_result.analysis_code}\n```\n\n")

        # Execute code
        print(f"[Iteration {iteration}] Executing analysis code...")
        success, output = execute_analysis_code(code_result.analysis_code)

        # Log execution result
        with open(discussion_path, 'a') as f:
            if success:
                f.write(f"**Execution:** Success\n\n")
                f.write(f"**Output:**\n```\n{output}\n```\n\n")
            else:
                f.write(f"**Execution:** Failed\n\n")
                f.write(f"**Error:**\n```\n{output}\n```\n\n")
            f.write("---\n\n")

        if success:
            print(f"[Iteration {iteration}] Execution successful!")
            break
        else:
            print(f"[Iteration {iteration}] Execution failed: {output[:200]}...")
            error_feedback = output

            if iteration == max_debug_iterations:
                print(f"\nFailed after {max_debug_iterations} attempts. Giving up.")
                return 2

    # Check if we have valid results (should always be true if we reach here)
    if code_result is None or output is None:
        print("\nError: No valid results from analysis loop")
        return 2

    # Generate report with image understanding
    print("\nAnalyzing visualizations with image understanding...")
    report_result = analyze_plots_with_vision(
        analysis_code=code_result.analysis_code,
        execution_output=output,
        dataset_config=dataset_config,
        problem_description=problem_description
    )

    # Save report
    report_path = os.path.join(DATA_ANALYSIS_DIR, "data_analysis_report.md")
    with open(report_path, 'w') as f:
        f.write(report_result.report_markdown)

    # Log total workflow timing

    print(f"\n{'='*80}")
    print("DATA ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"✓ Report saved: {report_path}")
    print(f"✓ Code saved: {os.path.join(DATA_ANALYSIS_DIR, 'analysis_code.py')}")
    print(f"✓ Log saved: {discussion_path}")
    print(f"✓ Plots saved: {os.path.join(DATA_ANALYSIS_DIR, 'plots/')}")
    print(f"{'='*80}\n")

    return 0
