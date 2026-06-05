"""
Telemetry module for AgenticSciML: records LLM token usage, costs, and execution timing.

Usage: set SCIML_TELEMETRY_DIR env var to enable. All records written to per-PID JSONL
files in that directory. Call merge_telemetry() at the end to produce summary JSON.
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
    """
    model_lower = model.lower()
    for key in _COST_LOOKUP_ORDER:
        if key in model_lower:
            in_rate, out_rate = COST_TABLE[key]
            in_cost = (input_tokens / 1_000_000) * in_rate
            out_cost = (output_tokens / 1_000_000) * out_rate
            return in_cost, out_cost, in_cost + out_cost
    # Unknown model - return 0 with warning
    return 0.0, 0.0, 0.0


# ============================================================================
# Record dataclasses
# ============================================================================

@dataclass
class LLMCallRecord:
    """Record for a single LLM API call."""
    record_type: str = "llm_call"
    timestamp: str = ""
    agent_role: str = ""
    model: str = ""
    iteration: int = -1
    solution_id: str = ""
    pid: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    input_cost_usd: float = 0.0
    output_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    latency_seconds: float = 0.0
    throughput_tps: float = 0.0
    token_source: str = ""     # which extraction path provided tokens


@dataclass
class ExecutionRecord:
    """Record for a training or evaluation subprocess execution."""
    record_type: str = "execution"
    timestamp: str = ""
    solution_id: str = ""
    iteration: int = -1
    stage: str = ""            # "validate", "train", or "evaluate"
    gpu_id: str = ""
    duration_seconds: float = 0.0
    status: str = ""           # "success" or "error" or "timeout"
    exit_code: int = -1


# ============================================================================
# File I/O
# ============================================================================

# Thread lock for writes within the same process
_write_lock = threading.Lock()


def _get_tel_file(telemetry_dir: str) -> str:
    """Return per-PID JSONL file path."""
    return os.path.join(telemetry_dir, f"tel_{os.getpid()}.jsonl")


def write_llm_record(record: LLMCallRecord, telemetry_dir: str) -> None:
    """Append an LLMCallRecord as a JSON line to the per-PID telemetry file."""
    os.makedirs(telemetry_dir, exist_ok=True)
    line = json.dumps(asdict(record)) + "\n"
    path = _get_tel_file(telemetry_dir)
    with _write_lock:
        with open(path, "a") as f:
            f.write(line)


def write_execution_record(record: ExecutionRecord, telemetry_dir: str) -> None:
    """Append an ExecutionRecord as a JSON line to the per-PID telemetry file."""
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
    """

    def __init__(self, agent_role: str, iteration: int, solution_id: str, telemetry_dir: str):
        super().__init__()
        self.agent_role = agent_role
        self.iteration = iteration
        self.solution_id = solution_id
        self.telemetry_dir = telemetry_dir
        self._start_time: float = 0.0
        self._model: str = ""

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs) -> None:
        self._start_time = time.time()
        # Try to extract model name from serialized metadata
        self._model = (
            serialized.get("kwargs", {}).get("model", "")
            or serialized.get("kwargs", {}).get("model_name", "")
            or serialized.get("name", "")
        )

    def on_chat_model_start(self, serialized: dict[str, Any], messages: list, **kwargs) -> None:
        """Called instead of on_llm_start for chat models."""
        self._start_time = time.time()
        self._model = (
            serialized.get("kwargs", {}).get("model", "")
            or serialized.get("kwargs", {}).get("model_name", "")
            or serialized.get("name", "")
        )

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        latency = time.time() - self._start_time

        # Extract token usage — try multiple paths for cross-provider compatibility
        input_tokens = 0
        output_tokens = 0
        token_source = "unknown"

        # Path 1: Standard LangChain usage_metadata on the AIMessage
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

        # Path 2: llm_output dict (OpenAI legacy, some providers)
        if token_source == "unknown" and response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            if usage:
                input_tokens = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0) or 0
                output_tokens = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0) or 0
                token_source = "llm_output.token_usage"

        # Path 3: generation_info (Gemini)
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

        # Compute cost
        in_cost, out_cost, total_cost = compute_cost(self._model, input_tokens, output_tokens)

        # Throughput
        total_tokens = input_tokens + output_tokens
        throughput = total_tokens / latency if latency > 0 else 0.0

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
    """
    if not os.path.isdir(telemetry_dir):
        print(f"[telemetry] No telemetry directory found: {telemetry_dir}")
        return

    import glob

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

    # Sort by timestamp
    all_records.sort(key=lambda r: r.get("timestamp", ""))

    # Write consolidated calls JSONL
    calls_path = os.path.join(os.path.dirname(output_summary_path), "telemetry_calls.jsonl")
    with open(calls_path, "w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")

    # Separate record types
    llm_records = [r for r in all_records if r.get("record_type") == "llm_call"]
    exec_records = [r for r in all_records if r.get("record_type") == "execution"]

    # LLM summary
    total_input_tokens = sum(r.get("input_tokens", 0) for r in llm_records)
    total_output_tokens = sum(r.get("output_tokens", 0) for r in llm_records)
    total_cost = sum(r.get("total_cost_usd", 0.0) for r in llm_records)
    total_llm_latency = sum(r.get("latency_seconds", 0.0) for r in llm_records)

    # Breakdown by agent_role
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

    # Breakdown by model
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

    # Breakdown by iteration
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

    # Execution summary
    train_records = [r for r in exec_records if r.get("stage") == "train"]
    eval_records = [r for r in exec_records if r.get("stage") == "evaluate"]
    validate_records = [r for r in exec_records if r.get("stage") == "validate"]

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

    with open(output_summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[telemetry] Summary written to {output_summary_path}")
    print(f"[telemetry] Total LLM cost: ${summary['llm_summary']['total_cost_usd']:.4f} "
          f"| Tokens: {summary['llm_summary']['total_tokens']:,} "
          f"| Calls: {summary['llm_summary']['total_calls']}")
