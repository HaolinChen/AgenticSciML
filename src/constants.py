"""
Configuration constants for the SciML Agent system.

============================================================================
中文模块说明（学习注释）
============================================================================
作用：
    全系统的集中配置模块。定义 LLM 模型选择与路由、各智能体使用的模型与采样
    温度、目录与文件路径、进化/选择/调试的各类上限与并发参数、以及各类执行超时。
    几乎所有配置都可通过 SCIML_* 环境变量覆盖，未设置则使用此处的默认值。

在 pipeline 中的位置：
    被 main.py 及各阶段模块（create_contract/create_root/propose_critic/
    select_mutations/analyze/retrieve_* 等）导入，作为“唯一事实来源”的配置。

主要输入：
    环境变量（均为可选，前缀 SCIML_）。例如 SCIML_USE_MINI、SCIML_MODEL_*、
    SCIML_AGENT_MODEL_*、SCIML_TEMP_*、SCIML_*_DIR、SCIML_MAX_EVOLUTIONARY_ITERATIONS、
    SCIML_MUTATION_BATCH、SCIML_SELECTION_POOL_SIZE、SCIML_ENSEMBLE_MODELS 等。

主要输出（供其它模块导入的模块级常量）：
    - 模型/智能体：MODELS、AGENT_MODELS、TEMPERATURES、ENSEMBLE_MODELS、USE_MINI
    - 目录：USER_INPUT_DIR、TESTING_DIR、KB_DIR、AB_DIR、PROPOSAL_POOL_DIR、
      SOLUTION_AND_OUTPUTS_DIR、RESULTS_DIR、DATA_ANALYSIS_DIR、SELECTION_LOGS_DIR
    - 文件：RESULTS_FILE、KB_INDEX_FILE、DATASET_CONFIG_PATH
    - 上限/并发：MAX_DEBUG_ITERATIONS、MAX_REFINEMENT_ITERATIONS、
      MAX_PROPOSE_CRITIC_ROUNDS、MAX_CHILDREN_PER_NODE、MAX_EVOLUTIONARY_ITERATIONS、
      MUTATION_BATCH、SELECTION_POOL_SIZE
    - 超时：TIMEOUT_VALIDATION、TIMEOUT_TRAINING、TIMEOUT_EVALUATION、TIMEOUT_DATA_ANALYSIS

关键函数列表：
    - _env_str/_env_int/_env_float/_env_bool/_env_list  从环境变量读取并按类型解析（带默认值）
    - get_speaker_name(agent_role)  根据智能体所用模型推断服务商，生成对话发言人标识
============================================================================
"""

import os


# 下面五个 _env_* 辅助函数：统一从环境变量读取配置并按目标类型解析；
# 环境变量未设置时返回传入的 default。这样每个常量都可被 SCIML_* 环境变量覆盖。
def _env_str(name: str, default: str) -> str:
    """中文：读环境变量为字符串，未设置则用 default。"""
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    """中文：读环境变量并转 int，未设置则用 default。"""
    value = os.getenv(name)
    return int(value) if value is not None else default


def _env_float(name: str, default: float) -> float:
    """中文：读环境变量并转 float，未设置则用 default。"""
    value = os.getenv(name)
    return float(value) if value is not None else default


def _env_bool(name: str, default: bool) -> bool:
    """中文：读环境变量并转 bool；识别 1/true/yes/on（大小写不敏感）为真，未设置用 default。"""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: list[str]) -> list[str]:
    """中文：读环境变量并按逗号切分为去空白的字符串列表，未设置用 default。"""
    value = os.getenv(name)
    if value is None:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]

# ============================================================================
# LLM Configuration
# ============================================================================



# Available LLM models
USE_MINI = _env_bool("SCIML_USE_MINI", True)  # Mini versions are usually sufficient and cheaper
MODELS = {
    "gpt": _env_str("SCIML_MODEL_GPT", "gpt-5-mini" if USE_MINI else "gpt-5"),
    "claude": _env_str("SCIML_MODEL_CLAUDE", "claude-haiku-4-5" if USE_MINI else "claude-sonnet-4-5"),
    "gemini": _env_str("SCIML_MODEL_GEMINI", "gemini-2.5-flash" if USE_MINI else "gemini-2.5-pro"),
    "grok": _env_str("SCIML_MODEL_GROK", "grok-4-fast-reasoning" if USE_MINI else "grok-4")
}

# Agent-to-Model assignments
AGENT_MODELS = {
    "tester": _env_str("SCIML_AGENT_MODEL_TESTER", MODELS["claude"]),              # Contract generation / evaluator
    "root_engineer": _env_str("SCIML_AGENT_MODEL_ROOT_ENGINEER", MODELS["claude"]),# Root solution code generation
    "engineer": _env_str("SCIML_AGENT_MODEL_ENGINEER", MODELS["claude"]),           # Code generation
    "analyst": _env_str("SCIML_AGENT_MODEL_ANALYST", MODELS["gemini"]),             # Result analysis
    "proposer": _env_str("SCIML_AGENT_MODEL_PROPOSER", MODELS["gemini"]),           # Proposal generation
    "critic": _env_str("SCIML_AGENT_MODEL_CRITIC", MODELS["gpt"]),                  # Critical evaluation
    "retriever": _env_str("SCIML_AGENT_MODEL_RETRIEVER", MODELS["gemini"]),         # Knowledge base retrieval
    "debugger": _env_str("SCIML_AGENT_MODEL_DEBUGGER", MODELS["gpt"]),              # Debugging assistance
    "data_analyst": _env_str("SCIML_AGENT_MODEL_DATA_ANALYST", MODELS["gemini"])    # Data analysis
}

# Temperature settings for different agent types
TEMPERATURES = {
    "tester": _env_float("SCIML_TEMP_TESTER", 0.0),                # Deterministic for contract generation
    "root_engineer": _env_float("SCIML_TEMP_ROOT_ENGINEER", 0.0),  # Deterministic for root solution code generation
    "engineer": _env_float("SCIML_TEMP_ENGINEER", 0.0),            # Deterministic for code generation
    "analyst": _env_float("SCIML_TEMP_ANALYST", 0.1),              # Result analysis
    "proposer": _env_float("SCIML_TEMP_PROPOSER", 0.5),            # Proposal generation
    "critic": _env_float("SCIML_TEMP_CRITIC", 0.3),                # Critical evaluation
    "retriever": _env_float("SCIML_TEMP_RETRIEVER", 0.0),          # Deterministic for retrieval
    "debugger": _env_float("SCIML_TEMP_DEBUGGER", 0.0),            # Deterministic for debugging
    "data_analyst": _env_float("SCIML_TEMP_DATA_ANALYST", 0.3)     # Data analysis
}

# ============================================================================
# Helper Functions
# ============================================================================

def get_speaker_name(agent_role: str) -> str:
    """
    Get speaker identifier for conversation history based on agent's assigned model.

    Args:
        agent_role: Role name (e.g., "proposer", "critic", "analyst")

    Returns:
        Speaker name formatted as "{role}_{provider}" (e.g., "proposer_gemini", "critic_gpt")

    中文：根据智能体角色查出其绑定模型，再从模型名中推断服务商（gpt/gemini/claude/grok），
        拼成对话历史里的“发言人”标识 "{角色}_{服务商}"。识别不到服务商时用 "unknown"。
    """
    model = AGENT_MODELS.get(agent_role, "")

    # Extract provider from model string
    model_lower = model.lower()
    if "gpt" in model_lower:
        provider = "gpt"
    elif "gemini" in model_lower:
        provider = "gemini"
    elif "claude" in model_lower:
        provider = "claude"
    elif "grok" in model_lower:
        provider = "grok"
    else:
        # Fallback to just the role name if provider unknown
        provider = "unknown"

    return f"{agent_role}_{provider}"

# ============================================================================
# Directory Paths (relative to src/)
# ============================================================================

USER_INPUT_DIR = _env_str("SCIML_USER_INPUT_DIR", "./USER_INPUT")
TESTING_DIR = _env_str("SCIML_TESTING_DIR", "./TESTING")
KB_DIR = _env_str("SCIML_KB_DIR", "./KB")
AB_DIR = _env_str("SCIML_AB_DIR", "./AB")
PROPOSAL_POOL_DIR = _env_str("SCIML_PROPOSAL_POOL_DIR", "./PROPOSAL_POOL")
SOLUTION_AND_OUTPUTS_DIR = _env_str("SCIML_SOLUTION_AND_OUTPUTS_DIR", "./SOLUTION_AND_OUTPUTS")
RESULTS_DIR = _env_str("SCIML_RESULTS_DIR", "./RESULTS")
DATA_ANALYSIS_DIR = _env_str("SCIML_DATA_ANALYSIS_DIR", "./DATA_ANALYSIS")

# ============================================================================
# File Paths
# ============================================================================

RESULTS_FILE = f"{RESULTS_DIR}/results.json"
KB_INDEX_FILE = f"{KB_DIR}/indices.json"
DATASET_CONFIG_PATH = f"{USER_INPUT_DIR}/dataset_config.json"

# ============================================================================
# Evaluation Settings
# ============================================================================

# Maximum number of bug-fix iterations in engineer-execute-debug loop
MAX_DEBUG_ITERATIONS = 5

# Maximum number of refinement iterations in human-in-the-loop approval
MAX_REFINEMENT_ITERATIONS = 5

# Maximum number of (Propose, Critic) rounds in proposal generation
# Rounds 1 to N-2: Propose (reasoning) → Critique (reasoning) → Refine
# Round N-1: Propose (synthesis) → Critique (plan) → Refine
# Round N: Finalization (no critique)
MAX_PROPOSE_CRITIC_ROUNDS = 3

# Maximum number of children per solution node
# When a solution reaches this limit, it's skipped during champion selection
MAX_CHILDREN_PER_NODE = 10

# Maximum number of evolutionary iterations in Phase 3
MAX_EVOLUTIONARY_ITERATIONS = _env_int("SCIML_MAX_EVOLUTIONARY_ITERATIONS", 8)

# ============================================================================
# Mutation and Selection Configuration (Ensemble-Guided Batch Mutation)
# ============================================================================

# Number of parallel mutations per iteration
MUTATION_BATCH = _env_int("SCIML_MUTATION_BATCH", 4)

# Top-K solutions for selector ensemble to review
SELECTION_POOL_SIZE = _env_int("SCIML_SELECTION_POOL_SIZE", 8)

# Ensemble models for selection voting
ENSEMBLE_MODELS = _env_list("SCIML_ENSEMBLE_MODELS", ["gpt", "grok", "gemini"])

# Selection logging directory
SELECTION_LOGS_DIR = os.environ.get("SCIML_SELECTION_LOGS_DIR", "./SELECTION_LOGS")

# ============================================================================
# Execution Timeouts (in seconds)
# ============================================================================

# Timeout for validation mode (fast, 1 epoch smoke test)
TIMEOUT_VALIDATION = 600  # 10 minutes

# Timeout for training mode (full training run)
TIMEOUT_TRAINING = 7200  # 2 hours

# Timeout for evaluation script
TIMEOUT_EVALUATION = 600  # 10 minutes

# Timeout for data analysis execution
TIMEOUT_DATA_ANALYSIS = 600  # 10 minutes
