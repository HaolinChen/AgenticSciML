"""
Telemetry module for AgenticSciML: records LLM token usage, costs, and execution timing.

Usage: set SCIML_TELEMETRY_DIR env var to enable. All records written to per-PID JSONL
files in that directory. Call merge_telemetry() at the end to produce summary JSON.

================================ 中文模块说明 ================================
【作用】
    遥测(telemetry)工具模块。统计每次 LLM 调用的 token 用量、美元成本、耗时/吞吐，
    以及训练/评估等子进程执行的时长与状态；把明细以 JSONL 追加落盘，最后可汇总为
    一份 telemetry_summary.json。

【所属角色】
    横切基础设施。被 agents._llm_invoke（通过 LangChain 回调采集 LLM 调用）与
    main.py（在流程结束时调用 merge_telemetry 汇总）使用。

【主要输入】
    - 环境变量：SCIML_TELEMETRY_DIR —— 遥测目录；由调用方读取后作为参数传入本模块
      （本模块自身不直接读该环境变量，只接收 telemetry_dir 参数）。
    - 运行期数据：LangChain 的 LLMResult（token 用量来源）、模型名、agent 角色、
      迭代号、solution_id 等。
    - 成本表：COST_TABLE（各模型每百万 token 的输入/输出单价，单位 USD）。

【主要输出】
    - 明细文件：{telemetry_dir}/tel_{PID}.jsonl —— 每进程一份，逐行 JSON 记录。
    - 汇总文件（merge_telemetry 产出）：
        * telemetry_summary.json —— 总量与多维度(角色/模型/迭代)成本、执行统计。
        * telemetry_calls.jsonl  —— 合并全部进程、按时间排序的明细。

【关键函数/类清单】
    - compute_cost(...)         : 依据模型名与 token 数计算美元成本。
    - LLMCallRecord / ExecutionRecord : 两类记录的 dataclass 数据结构。
    - write_llm_record / write_execution_record : 线程安全地追加写 JSONL。
    - TelemetryCallback         : LangChain 回调，采集单次 LLM 调用的 token 与延迟。
    - merge_telemetry(...)      : 汇总所有 JSONL，产出 summary 与合并明细。
============================================================================
"""

import os
import json
import time
import threading
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


# ============================================================================
# Cost table (USD per 1M tokens, uncached, unbatched) — from api.md
# ============================================================================

# Keys: substring matched against model name (case-insensitive)
# Values: (input_cost_per_1m, output_cost_per_1m) in USD
# 中文：计价表。键为模型名的“子串”（大小写不敏感匹配），值为
#       (每百万输入 token 单价, 每百万输出 token 单价)，单位美元。
COST_TABLE = {
    "claude-haiku-4-5":       (1.00,   5.00),
    "claude-haiku-3-5":       (0.80,   4.00),
    "claude-sonnet-4-5":      (3.00,  15.00),
    "claude-opus-4-5":        (5.00,  25.00),
    "gpt-5-mini":             (0.25,   2.00),
    "gpt-5-nano":             (0.05,   0.40),
    "gpt-5":                  (1.25,  10.00),   # catches gpt-5 (not mini/nano)
    "gpt-4.1-mini":           (0.40,   1.60),
    "gpt-4.1-nano":           (0.10,   0.40),
    "gpt-4o-mini":            (0.15,   0.60),
    "gpt-4.1":                (2.00,   8.00),
    "gpt-4o":                 (2.50,  10.00),
    "gemini-2.5-flash":       (0.30,   2.50),
    "gemini-2.5-pro":         (1.25,  10.00),
    "grok-4-fast":            (0.20,   0.50),   # covers both reasoning and non-reasoning
    "grok-4-0709":            (3.00,  15.00),
    "grok-3-mini":            (0.30,   0.50),
    "grok-3":                 (3.00,  15.00),
}

# Lookup order matters: more specific substrings first
# 中文：查表顺序很重要——更具体的子串必须排在前面，避免被更宽泛的子串误命中。
#       例如 "gpt-5-mini"/"gpt-5-nano" 必须先于 "gpt-5" 匹配。
_COST_LOOKUP_ORDER = [
    "claude-haiku-4-5", "claude-haiku-3-5", "claude-sonnet-4-5", "claude-opus-4-5",
    "gpt-5-mini", "gpt-5-nano", "gpt-5",
    "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o-mini", "gpt-4.1", "gpt-4o",
    "gemini-2.5-flash", "gemini-2.5-pro",
    "grok-4-fast", "grok-4-0709", "grok-3-mini", "grok-3",
]


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> tuple[float, float, float]:
    """
    Compute USD cost for an LLM call.

    Returns:
        (input_cost_usd, output_cost_usd, total_cost_usd)

    中文说明：
        做什么：按模型名在 COST_TABLE 中查单价，换算本次调用的美元成本。
        参数：model——模型名；input_tokens/output_tokens——输入/输出 token 数。
        返回：(输入成本, 输出成本, 总成本)，单位美元。
        副作用：无（纯计算）。未知模型返回 (0,0,0)。
    """
    model_lower = model.lower()
    # 按预设顺序做子串匹配：命中即用其单价换算（token 数 / 100 万 × 单价）
    for key in _COST_LOOKUP_ORDER:
        if key in model_lower:
            in_rate, out_rate = COST_TABLE[key]
            in_cost = (input_tokens / 1_000_000) * in_rate
            out_cost = (output_tokens / 1_000_000) * out_rate
            return in_cost, out_cost, in_cost + out_cost
    # 未在计价表中的模型：成本按 0 计（避免报错，但会低估花费）
    return 0.0, 0.0, 0.0


# ============================================================================
# Record dataclasses
# ============================================================================

@dataclass
class LLMCallRecord:
    """Record for a single LLM API call.

    中文：单次 LLM 调用的遥测记录（asdict 后逐行写入 JSONL）。字段含义见行内注释。
    """
    record_type: str = "llm_call"      # 记录类型标记，用于 merge 时区分
    timestamp: str = ""                # UTC ISO 时间戳
    agent_role: str = ""               # 发起调用的智能体角色
    model: str = ""                    # 模型名
    iteration: int = -1                # 所属进化迭代号
    solution_id: str = ""              # 所属方案 ID
    pid: int = 0                       # 进程 PID
    input_tokens: int = 0              # 输入 token 数
    output_tokens: int = 0             # 输出 token 数
    input_cost_usd: float = 0.0        # 输入成本(USD)
    output_cost_usd: float = 0.0       # 输出成本(USD)
    total_cost_usd: float = 0.0        # 总成本(USD)
    latency_seconds: float = 0.0       # 本次调用耗时(秒)
    throughput_tps: float = 0.0        # 吞吐(tokens/秒)
    token_source: str = ""     # which extraction path provided tokens
                               # 中文：token 数取自哪条解析路径（便于排查计数问题）


@dataclass
class ExecutionRecord:
    """Record for a training or evaluation subprocess execution.

    中文：训练/评估等子进程执行的遥测记录。
    """
    record_type: str = "execution"     # 记录类型标记
    timestamp: str = ""                # UTC ISO 时间戳
    solution_id: str = ""              # 所属方案 ID
    iteration: int = -1                # 所属迭代号
    stage: str = ""            # "validate", "train", or "evaluate"
                               # 中文：阶段——校验/训练/评估
    gpu_id: str = ""                   # 使用的 GPU 编号
    duration_seconds: float = 0.0      # 执行时长(秒)
    status: str = ""           # "success" or "error" or "timeout"
                               # 中文：执行状态——成功/出错/超时
    exit_code: int = -1               # 子进程退出码


# ============================================================================
# File I/O
# ============================================================================

# Thread lock for writes within the same process
# 中文：同一进程内多线程写同一文件时的互斥锁（跨进程则靠“每进程一份文件”隔离）。
_write_lock = threading.Lock()


def _get_tel_file(telemetry_dir: str) -> str:
    """Return per-PID JSONL file path.

    中文：返回本进程专属的 JSONL 文件路径（以 PID 命名，避免多进程写冲突）。
    """
    return os.path.join(telemetry_dir, f"tel_{os.getpid()}.jsonl")


def write_llm_record(record: LLMCallRecord, telemetry_dir: str) -> None:
    """Append an LLMCallRecord as a JSON line to the per-PID telemetry file.

    中文：把一条 LLM 调用记录以 JSON 行追加写入本进程的遥测文件（加锁保证线程安全）。
    """
    os.makedirs(telemetry_dir, exist_ok=True)
    line = json.dumps(asdict(record)) + "\n"
    path = _get_tel_file(telemetry_dir)
    with _write_lock:
        with open(path, "a") as f:
            f.write(line)


def write_execution_record(record: ExecutionRecord, telemetry_dir: str) -> None:
    """Append an ExecutionRecord as a JSON line to the per-PID telemetry file.

    中文：把一条子进程执行记录以 JSON 行追加写入本进程的遥测文件（加锁保证线程安全）。
    """
    os.makedirs(telemetry_dir, exist_ok=True)
    line = json.dumps(asdict(record)) + "\n"
    path = _get_tel_file(telemetry_dir)
    with _write_lock:
        with open(path, "a") as f:
            f.write(line)


# ============================================================================
# LangChain callback
# ============================================================================

class TelemetryCallback(BaseCallbackHandler):
    """
    LangChain callback that captures token usage and latency for every LLM call.
    Attach to LLM constructors via callbacks=[TelemetryCallback(...)].

    中文：LangChain 回调处理器，为每次 LLM 调用自动采集 token 用量与耗时。
        使用方式：在构造 LLM 时传 callbacks=[TelemetryCallback(...)]。
        工作机制：on_(chat_model_)start 记开始时间/模型名 → on_llm_end 计算延迟、
                  抽取 token、算成本，并写一条 LLMCallRecord。
    """

    def __init__(self, agent_role: str, iteration: int, solution_id: str, telemetry_dir: str):
        # 保存本次调用的上下文标签（角色/迭代/方案/遥测目录），供落盘时写入记录
        super().__init__()
        self.agent_role = agent_role
        self.iteration = iteration
        self.solution_id = solution_id
        self.telemetry_dir = telemetry_dir
        self._start_time: float = 0.0
        self._model: str = ""

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs) -> None:
        # 非 chat 模型调用开始：记录起始时间，并尽力从序列化元数据里提取模型名
        self._start_time = time.time()
        self._model = (
            serialized.get("kwargs", {}).get("model", "")
            or serialized.get("kwargs", {}).get("model_name", "")
            or serialized.get("name", "")
        )

    def on_chat_model_start(self, serialized: dict[str, Any], messages: list, **kwargs) -> None:
        """Called instead of on_llm_start for chat models.

        中文：对话(chat)模型会触发此回调而非 on_llm_start，逻辑相同（记时间+取模型名）。
        """
        self._start_time = time.time()
        self._model = (
            serialized.get("kwargs", {}).get("model", "")
            or serialized.get("kwargs", {}).get("model_name", "")
            or serialized.get("name", "")
        )

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        # 调用结束：先算延迟，再抽取 token、算成本，最后落盘一条记录
        latency = time.time() - self._start_time

        # 抽取 token 用量 —— 不同 provider 字段各异，按三条路径依次兜底
        input_tokens = 0
        output_tokens = 0
        token_source = "unknown"

        # 路径1：LangChain 标准 usage_metadata（挂在 AIMessage 上）
        try:
            gen = response.generations[0][0]
            msg = getattr(gen, "message", None)
            if msg is not None:
                usage = getattr(msg, "usage_metadata", None)
                if usage:
                    input_tokens = usage.get("input_tokens", 0) or 0
                    output_tokens = usage.get("output_tokens", 0) or 0
                    token_source = "usage_metadata"
        except (IndexError, AttributeError):
            pass

        # 路径2：llm_output 字典（OpenAI 旧格式及部分 provider）
        if token_source == "unknown" and response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            if usage:
                input_tokens = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0) or 0
                output_tokens = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0) or 0
                token_source = "llm_output.token_usage"

        # 路径3：generation_info（Gemini 的 token 计数在此）
        if token_source == "unknown":
            try:
                gen_info = getattr(response.generations[0][0], "generation_info", None) or {}
                usage = gen_info.get("usage_metadata", {})
                if usage:
                    input_tokens = usage.get("prompt_token_count", 0) or 0
                    output_tokens = usage.get("candidates_token_count", 0) or 0
                    token_source = "generation_info"
            except (IndexError, AttributeError):
                pass

        # 依模型与 token 数计算成本
        in_cost, out_cost, total_cost = compute_cost(self._model, input_tokens, output_tokens)

        # 计算吞吐（tokens/秒）；延迟为 0 时避免除零
        total_tokens = input_tokens + output_tokens
        throughput = total_tokens / latency if latency > 0 else 0.0

        # 组装记录并落盘（写入本进程的 tel_{PID}.jsonl）
        record = LLMCallRecord(
            timestamp=datetime.utcnow().isoformat(),
            agent_role=self.agent_role,
            model=self._model,
            iteration=self.iteration,
            solution_id=self.solution_id,
            pid=os.getpid(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost_usd=in_cost,
            output_cost_usd=out_cost,
            total_cost_usd=total_cost,
            latency_seconds=round(latency, 3),
            throughput_tps=round(throughput, 1),
            token_source=token_source,
        )
        write_llm_record(record, self.telemetry_dir)


# ============================================================================
# Merge / summarize telemetry from all JSONL files
# ============================================================================

def merge_telemetry(telemetry_dir: str, output_summary_path: str) -> None:
    """
    Read all tel_*.jsonl files in telemetry_dir, aggregate, write summary JSON
    and a consolidated calls JSONL.

    中文说明：
        做什么：读取 telemetry_dir 下所有 tel_*.jsonl（各进程明细），按时间排序合并，
                并按“角色/模型/迭代”多维聚合 LLM 成本、汇总执行统计。
        参数：telemetry_dir——明细目录；output_summary_path——汇总 JSON 输出路径。
        返回：None。
        副作用：写 telemetry_summary.json（output_summary_path）与同目录下的
                telemetry_calls.jsonl（合并明细）；并打印总成本/token/调用数概览。
    """
    if not os.path.isdir(telemetry_dir):
        print(f"[telemetry] No telemetry directory found: {telemetry_dir}")
        return

    import glob

    # 逐个读取每进程 JSONL，逐行解析为 dict（跳过损坏行）
    all_records = []
    jsonl_files = glob.glob(os.path.join(telemetry_dir, "tel_*.jsonl"))
    for path in jsonl_files:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        all_records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    if not all_records:
        print("[telemetry] No records found.")
        return

    # 按时间戳排序，得到全局时间顺序的明细
    all_records.sort(key=lambda r: r.get("timestamp", ""))

    # 写出合并后的明细 telemetry_calls.jsonl（与 summary 同目录）
    calls_path = os.path.join(os.path.dirname(output_summary_path), "telemetry_calls.jsonl")
    with open(calls_path, "w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")

    # 按 record_type 分离两类记录
    llm_records = [r for r in all_records if r.get("record_type") == "llm_call"]
    exec_records = [r for r in all_records if r.get("record_type") == "execution"]

    # LLM 总量统计：总输入/输出 token、总成本、总延迟
    total_input_tokens = sum(r.get("input_tokens", 0) for r in llm_records)
    total_output_tokens = sum(r.get("output_tokens", 0) for r in llm_records)
    total_cost = sum(r.get("total_cost_usd", 0.0) for r in llm_records)
    total_llm_latency = sum(r.get("latency_seconds", 0.0) for r in llm_records)

    # 维度一：按智能体角色(agent_role)聚合调用数/token/成本/延迟
    cost_by_role: dict = {}
    for r in llm_records:
        role = r.get("agent_role", "unknown")
        if role not in cost_by_role:
            cost_by_role[role] = {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                                   "total_cost_usd": 0.0, "latency_seconds": 0.0}
        cost_by_role[role]["calls"] += 1
        cost_by_role[role]["input_tokens"] += r.get("input_tokens", 0)
        cost_by_role[role]["output_tokens"] += r.get("output_tokens", 0)
        cost_by_role[role]["total_cost_usd"] += r.get("total_cost_usd", 0.0)
        cost_by_role[role]["latency_seconds"] += r.get("latency_seconds", 0.0)

    # 维度二：按模型(model)聚合调用数/token/成本
    cost_by_model: dict = {}
    for r in llm_records:
        model = r.get("model", "unknown")
        if model not in cost_by_model:
            cost_by_model[model] = {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                                     "total_cost_usd": 0.0}
        cost_by_model[model]["calls"] += 1
        cost_by_model[model]["input_tokens"] += r.get("input_tokens", 0)
        cost_by_model[model]["output_tokens"] += r.get("output_tokens", 0)
        cost_by_model[model]["total_cost_usd"] += r.get("total_cost_usd", 0.0)

    # 维度三：按进化迭代号(iteration)聚合调用数/token/成本
    cost_by_iteration: dict = {}
    for r in llm_records:
        it = str(r.get("iteration", -1))
        if it not in cost_by_iteration:
            cost_by_iteration[it] = {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                                      "total_cost_usd": 0.0}
        cost_by_iteration[it]["calls"] += 1
        cost_by_iteration[it]["input_tokens"] += r.get("input_tokens", 0)
        cost_by_iteration[it]["output_tokens"] += r.get("output_tokens", 0)
        cost_by_iteration[it]["total_cost_usd"] += r.get("total_cost_usd", 0.0)

    # 执行统计：按阶段(train/evaluate/validate)分组
    train_records = [r for r in exec_records if r.get("stage") == "train"]
    eval_records = [r for r in exec_records if r.get("stage") == "evaluate"]
    validate_records = [r for r in exec_records if r.get("stage") == "validate"]

    # 再按方案 ID 汇总各阶段的时长/状态/迭代号
    exec_by_solution: dict = {}
    for r in exec_records:
        sid = r.get("solution_id", "unknown")
        stage = r.get("stage", "unknown")
        if sid not in exec_by_solution:
            exec_by_solution[sid] = {}
        exec_by_solution[sid][stage] = {
            "duration_seconds": r.get("duration_seconds", 0.0),
            "status": r.get("status", "unknown"),
            "iteration": r.get("iteration", -1),
        }

    # 组装最终汇总结构（成本字段统一四舍五入到 4 位小数）
    summary = {
        "llm_summary": {
            "total_calls": len(llm_records),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "total_cost_usd": round(total_cost, 4),
            "total_latency_seconds": round(total_llm_latency, 2),
        },
        "cost_by_agent_role": {
            k: {**v, "total_cost_usd": round(v["total_cost_usd"], 4)}
            for k, v in sorted(cost_by_role.items())
        },
        "cost_by_model": {
            k: {**v, "total_cost_usd": round(v["total_cost_usd"], 4)}
            for k, v in sorted(cost_by_model.items())
        },
        "cost_by_iteration": {
            k: {**v, "total_cost_usd": round(v["total_cost_usd"], 4)}
            for k, v in sorted(cost_by_iteration.items(), key=lambda x: int(x[0]))
        },
        "execution_summary": {
            "total_train_runs": len(train_records),
            "total_eval_runs": len(eval_records),
            "total_validate_runs": len(validate_records),
            "total_train_seconds": round(sum(r.get("duration_seconds", 0) for r in train_records), 2),
            "total_eval_seconds": round(sum(r.get("duration_seconds", 0) for r in eval_records), 2),
            "total_validate_seconds": round(sum(r.get("duration_seconds", 0) for r in validate_records), 2),
        },
        "execution_by_solution": exec_by_solution,
    }

    # 写出汇总 JSON，并打印一行总览（成本/token/调用数）
    with open(output_summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[telemetry] Summary written to {output_summary_path}")
    print(f"[telemetry] Total LLM cost: ${summary['llm_summary']['total_cost_usd']:.4f} "
          f"| Tokens: {summary['llm_summary']['total_tokens']:,} "
          f"| Calls: {summary['llm_summary']['total_calls']}")
