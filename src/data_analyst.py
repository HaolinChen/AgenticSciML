"""
Data Analyst Agent module for exploratory data analysis.

This standalone module performs EDA on training datasets using a Gemini agent
with image understanding capabilities. It generates analysis code, executes it,
analyzes visualizations, and produces text-only reports for downstream agents.

================================ 中文模块说明 ================================
【作用】
    数据分析师(data_analyst)智能体模块，负责对用户训练集执行探索性数据分析(EDA)。
    它先让 LLM“生成 EDA 代码”，再以子进程“执行代码”产出图表与统计文本，
    然后用具备图像理解能力的 Gemini 模型“看图”写出纯文本分析报告，供下游只能读文本的
    智能体使用。

【所属阶段/角色】
    Phase 0.5（数据分析阶段）的核心；角色为 data_analyst。被 run_data_analysis.py 调用。

【主要输入】
    - 文件：
        * DATASET_CONFIG_PATH 指定的 dataset_config.json（内含 training_set：filename/
          description/loading_instructions，指向 .npz 训练集）。
        * USER_INPUT_DIR 下的问题描述文件：problem.md / requirements.md / evaluation.md。
        * 训练集 .npz（由生成的分析代码在 USER_INPUT_DIR 工作目录中加载）。
    - 参数：max_debug_iterations（生成-执行-调试循环的最大轮数）。
    - 环境变量：无直接读取（LLM 密钥/遥测目录由 constants/agents 层处理）。
    - 配置常量：AGENT_MODELS、TEMPERATURES、DATA_ANALYSIS_DIR、USER_INPUT_DIR、
      DATASET_CONFIG_PATH、TIMEOUT_DATA_ANALYSIS。

【主要输出】（均写入 DATA_ANALYSIS_DIR，即 DATA_ANALYSIS/）
    - analysis_code.py：最近一次生成并执行的 EDA 代码。
    - plots/：EDA 生成的图片(png/jpg)。
    - analysis_log.md：每轮生成/执行/报错的讨论日志。
    - data_analysis_report.md：最终纯文本分析报告（下游智能体读取）。
    - 返回值：run_analysis_workflow 返回 0 成功 / 非 0 失败。

【关键函数清单】
    - generate_analysis_code(...)   : 调 LLM 生成/修复 EDA 代码（结构化输出）。
    - execute_analysis_code(...)    : 落盘并以子进程执行 EDA 代码，捕获输出/错误。
    - analyze_plots_with_vision(...): 用 Gemini 图像理解，把图表转写为纯文本报告。
    - run_analysis_workflow(...)    : 主工作流，含“生成-执行-调试”循环并产出报告。
============================================================================
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
    """Structured output for data analyst code generation

    中文：data_analyst 生成 EDA 代码时的结构化输出 schema。
    - analysis_code：完整可执行的 EDA Python 代码。
    - analysis_plan：本次分析的目的与内容简述（用于日志记录）。
    """
    analysis_code: str = Field(
        description="Complete Python code for exploratory data analysis"
    )
    analysis_plan: str = Field(
        description="Brief description of what this code will analyze and why"
    )


class AnalysisReportOutput(BaseModel):
    """Structured output for data analyst report generation

    中文：data_analyst 生成分析报告时的结构化输出 schema。
    - report_markdown：仅含文本的 Markdown 报告，描述图表中观察到的发现。
    """
    report_markdown: str = Field(
        description="Text-only analysis report describing findings from visualizations"
    )


# ============================================================================
# System Prompts
# ============================================================================
# 中文：以下两个 system prompt 为英文提示词字面量，严禁改动其内容。
#   - ANALYST_CODE_GENERATION_PROMPT：指导 LLM 生成“可执行的 EDA 代码”。
#   - ANALYST_REPORT_GENERATION_PROMPT：指导具备视觉能力的 LLM“看图写纯文本报告”。

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

    中文说明：
        做什么：调用 data_analyst 角色的 LLM 生成 EDA 代码；若传入 error_feedback
                则进入“调试模式”，让 LLM 依据上次报错修复代码。
        参数：
            dataset_config      —— dataset_config.json 解析出的字典。
            problem_description —— 拼接后的问题描述文本。
            error_feedback      —— 上一次执行的报错（空串表示首次生成）。
            plots_dir_absolute  —— 图表保存目录的绝对路径（写入 prompt 供代码使用）。
        返回：AnalysisCodeOutput（analysis_code + analysis_plan）。
        副作用：发起一次 LLM 调用（结构化输出）；不写文件。
    """
    llm = get_llm("data_analyst")

    # 从 dataset_config 中取出训练集元信息（文件名/描述/加载说明），拼入 prompt
    training_set = dataset_config.get("training_set", {})
    filename = training_set.get("filename", "")
    description = training_set.get("description", "")
    loading_instructions = training_set.get("loading_instructions", "")

    # 构造用户消息：区分“调试模式”（带上次报错）与“首次生成”两种分支
    if error_feedback:
        # 调试模式：把上一轮报错反馈给 LLM，要求针对性修复
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
        # 首次生成：仅提供问题与数据集信息，让 LLM 从零生成 EDA 代码
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

    # 以结构化输出方式调用 LLM，强制返回 AnalysisCodeOutput 结构
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

    中文说明：
        做什么：把 LLM 生成的 EDA 代码落盘为 analysis_code.py，并以独立 Python
                子进程执行，捕获 stdout/stderr。
        参数：analysis_code——待执行代码；timeout——超时秒数（默认 TIMEOUT_DATA_ANALYSIS）。
        返回：(success, output)。成功时 output 为 stdout；失败时 output 为
              汇总的报错文本（含 exit code、stdout、stderr），供下一轮调试使用。
        副作用：写 DATA_ANALYSIS/analysis_code.py，创建 DATA_ANALYSIS/plots/ 目录，
                起子进程；子进程工作目录设为 USER_INPUT_DIR 以便访问数据集。
    """
    if timeout is None:
        timeout = TIMEOUT_DATA_ANALYSIS

    # 将代码写入固定文件（绝对路径），并预建 plots 输出目录
    code_path = os.path.abspath(os.path.join(DATA_ANALYSIS_DIR, "analysis_code.py"))
    os.makedirs(DATA_ANALYSIS_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATA_ANALYSIS_DIR, "plots"), exist_ok=True)

    with open(code_path, 'w') as f:
        f.write(analysis_code)

    # 以子进程执行代码（隔离运行，避免污染主进程；带超时保护）
    try:
        result = subprocess.run(
            [sys.executable, code_path],
            cwd=USER_INPUT_DIR,  # 在 USER_INPUT 目录执行，便于按相对路径访问数据集
            capture_output=True,
            text=True,
            timeout=timeout
        )

        if result.returncode == 0:
            return True, result.stdout
        else:
            # 非零退出：汇总退出码与标准输出/错误，作为调试反馈返回
            error_msg = f"Exit code {result.returncode}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
            return False, error_msg

    except subprocess.TimeoutExpired:
        # 超时：视为失败并返回超时说明
        return False, f"Execution timeout after {timeout} seconds"
    except Exception as e:
        # 其它异常：统一转为失败并返回异常信息
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

    中文说明：
        做什么：加载 DATA_ANALYSIS/plots/ 下的所有图片，连同代码与执行文本一起
                发给具备视觉能力的 Gemini 模型，产出“纯文本”分析报告。
        参数：analysis_code、execution_output（执行文本输出）、dataset_config、
              problem_description。
        返回：AnalysisReportOutput（report_markdown）。
        副作用：直接构造并调用 ChatGoogleGenerativeAI（Gemini，带视觉），读取图片文件；
                无本地文件写入（报告的落盘在 run_analysis_workflow 中完成）。
    """
    # 直接构造带视觉能力的 Gemini 模型（此处不走 get_llm，需自定义 max_output_tokens）
    model_name = AGENT_MODELS["data_analyst"]
    temperature = TEMPERATURES["data_analyst"]

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        max_output_tokens=16000
    )

    # 收集 DATA_ANALYSIS/plots/ 下的所有图片文件，排序以保证顺序稳定
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

    # 组装多模态消息：先放文本提示，再逐一追加 base64 编码的图片
    message_parts: list[MessageLikeRepresentation] = []

    # 追加文本部分（问题、数据集信息、执行输出、代码等上下文）
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

    # 逐张图片读入并转 base64，按 LangChain 的 image_url 格式追加到消息中
    for plot_path in plot_files:
        try:
            with open(plot_path, 'rb') as f:
                image_bytes = f.read()
                image_b64 = base64.b64encode(image_bytes).decode('utf-8')
                # 采用 data URI（内联 base64）方式传图，避免依赖外部可访问 URL
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

    # 结构化调用，强制返回仅含文本的 AnalysisReportOutput
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

    中文说明：
        做什么：Phase 0.5 的入口工作流。串联“生成代码→执行→(失败则)调试”循环，
                成功后再用视觉模型写报告并落盘。
        参数：max_debug_iterations——生成-执行-调试循环最大轮数。
        返回：0 成功；1 缺少配置/训练集；2 调试轮数用尽仍失败或结果无效。
        副作用：读 dataset_config.json 与 USER_INPUT 下问题文件；写 analysis_log.md、
                analysis_code.py、plots/、data_analysis_report.md；多次调用 LLM，起子进程。
    """

    print("\n" + "="*80)
    print("DATA ANALYSIS WORKFLOW")
    print("="*80)

    # 1) 读取 dataset_config.json（缺失或无 training_set 直接失败返回）
    if not os.path.exists(DATASET_CONFIG_PATH):
        print("Error: dataset_config.json not found")
        return 1

    with open(DATASET_CONFIG_PATH, 'r') as f:
        dataset_config = json.load(f)

    if "training_set" not in dataset_config:
        print("Error: No training_set in dataset_config.json")
        return 1

    # 2) 拼接问题描述：按顺序读取 USER_INPUT 下三个可选文件并合并
    problem_files = ["problem.md", "requirements.md", "evaluation.md"]
    problem_description = ""
    for fname in problem_files:
        fpath = os.path.join(USER_INPUT_DIR, fname)
        if os.path.exists(fpath):
            with open(fpath, 'r') as f:
                problem_description += f"\n\n## {fname}\n\n{f.read()}"

    # 3) 初始化讨论日志 analysis_log.md（写入表头，后续每轮追加）
    os.makedirs(DATA_ANALYSIS_DIR, exist_ok=True)
    discussion_path = os.path.join(DATA_ANALYSIS_DIR, "analysis_log.md")
    with open(discussion_path, 'w') as f:
        f.write("# Data Analysis Log\n\n")
        f.write(f"**Dataset:** {dataset_config['training_set']['filename']}\n\n")
        f.write("---\n\n")

    # 4) 计算 plots 目录绝对路径（写进 prompt，确保生成代码把图存到正确位置）
    plots_dir_absolute = os.path.abspath(os.path.join(DATA_ANALYSIS_DIR, "plots"))
    os.makedirs(plots_dir_absolute, exist_ok=True)

    # 5) 生成-执行-调试循环：error_feedback 在失败后携带上一轮报错反哺下一轮生成
    error_feedback = ""
    code_result = None
    output = None

    for iteration in range(1, max_debug_iterations + 1):
        print(f"\n[Iteration {iteration}/{max_debug_iterations}] Generating analysis code...")

        # 生成（或修复）EDA 代码
        code_result = generate_analysis_code(
            dataset_config=dataset_config,
            problem_description=problem_description,
            error_feedback=error_feedback,
            plots_dir_absolute=plots_dir_absolute
        )

        # Log timing for code generation

        # 把本轮计划/上次报错/生成代码追加到讨论日志
        with open(discussion_path, 'a') as f:
            f.write(f"## Iteration {iteration}\n\n")
            f.write(f"**Plan:** {code_result.analysis_plan}\n\n")
            if error_feedback:
                f.write(f"**Previous Error:**\n```\n{error_feedback}\n```\n\n")
            f.write(f"**Generated Code:**\n```python\n{code_result.analysis_code}\n```\n\n")

        # 执行 EDA 代码，拿到成功标志与输出/报错文本
        print(f"[Iteration {iteration}] Executing analysis code...")
        success, output = execute_analysis_code(code_result.analysis_code)

        # 把执行结果（成功输出或失败报错）追加到讨论日志
        with open(discussion_path, 'a') as f:
            if success:
                f.write(f"**Execution:** Success\n\n")
                f.write(f"**Output:**\n```\n{output}\n```\n\n")
            else:
                f.write(f"**Execution:** Failed\n\n")
                f.write(f"**Error:**\n```\n{output}\n```\n\n")
            f.write("---\n\n")

        if success:
            # 成功即跳出循环，进入报告生成阶段
            print(f"[Iteration {iteration}] Execution successful!")
            break
        else:
            # 失败：记录报错作为下一轮反馈；若已是最后一轮则放弃并返回 2
            print(f"[Iteration {iteration}] Execution failed: {output[:200]}...")
            error_feedback = output

            if iteration == max_debug_iterations:
                print(f"\nFailed after {max_debug_iterations} attempts. Giving up.")
                return 2

    # 兜底校验：正常情况下到此必有有效结果（防御性检查）
    if code_result is None or output is None:
        print("\nError: No valid results from analysis loop")
        return 2

    # 6) 用视觉模型看图生成纯文本报告
    print("\nAnalyzing visualizations with image understanding...")
    report_result = analyze_plots_with_vision(
        analysis_code=code_result.analysis_code,
        execution_output=output,
        dataset_config=dataset_config,
        problem_description=problem_description
    )

    # 7) 落盘最终报告 data_analysis_report.md（供下游智能体读取）
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
