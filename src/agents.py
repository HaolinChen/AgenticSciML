"""
Shared agent functions for SciML Agent system.

This module contains reusable agent functions that can be called from
multiple scripts (create_contract.py, create_root.py, etc.)
"""

from dotenv import load_dotenv
from typing import Literal, Any
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage, MessageLikeRepresentation
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from constants import AGENT_MODELS, TEMPERATURES, MAX_PROPOSE_CRITIC_ROUNDS, get_speaker_name

load_dotenv()

# ============================================================================
# Pydantic Models
# ============================================================================

class ContractOutput(BaseModel):
    """Structured output for Tester agent"""
    evaluate_py: str = Field(description="Complete Python code for evaluate.py")
    guidelines_md: str = Field(description="Complete markdown content for guidelines.md")


class SolutionOutput(BaseModel):
    """Structured output for Engineer agent"""
    solution_code: str = Field(description="Complete Python code for solution.py")


class ValidationDecision(BaseModel):
    """Structured output for Validator agent"""
    culprit: Literal["tester_error", "engineer_error", "success"] = Field(
        description="Who is responsible for the validation error"
    )
    specific_feedback: str = Field(
        description="Specific feedback on what to fix"
    )


class KBRetrievalResult(BaseModel):
    """Structured output for KB retriever agent"""
    selected_entry_index: int | None = Field(
        description="Index of the selected KB entry from indices.json (0-based), or None if no entry is relevant"
    )
    reasoning: str = Field(
        description="Detailed reasoning for why this entry was selected (or why none were selected)"
    )


class AnalysisReport(BaseModel):
    """Structured output for Analyst agent"""
    analysis_markdown: str = Field(
        description="Complete markdown analysis report with sections: Summary, Training Dynamics, Performance Breakdown, Problems Identified, Comparison with Parent (if applicable)"
    )
    plot_analysis: str | None = Field(
        default=None,
        description="Separate section analyzing plots if images were provided (optional). Should describe what the plots show and key observations."
    )


class DebugSuggestion(BaseModel):
    """Structured output for Debugger agent"""
    suggestion: str = Field(description="Concise, actionable debugging suggestion")


class ProposalOutput(BaseModel):
    """Structured output for Proposer agent"""
    proposal_markdown: str = Field(description="Complete proposal in markdown format")


class CritiqueOutput(BaseModel):
    """Structured output for Critic agent"""
    critique_markdown: str = Field(description="Complete critique with suggestions")


# ============================================================================
# LLM Initialization
# ============================================================================

def get_llm(agent_name: str):
    """Get LLM instance for a specific agent."""
    import os as _os
    model_name = AGENT_MODELS[agent_name]
    temperature = TEMPERATURES[agent_name]

    # This needs to be changed based on what model you are using
    max_tokens_setting = 60000

    if "claude" in model_name:
        return ChatAnthropic(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens_setting,
        )
    elif "gpt" in model_name or "o1" in model_name:
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens_setting,
        )
    elif "gemini" in model_name:
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_output_tokens=max_tokens_setting,
        )
    elif "grok" in model_name:
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens_setting,
            base_url="https://api.x.ai/v1",
            api_key=_os.environ.get("XAI_API_KEY"),
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def _llm_invoke(llm, schema, messages, agent_name: str):
    """
    Invoke an LLM with structured output. If SCIML_TELEMETRY_DIR is set,
    extract token usage and cost using include_raw=True and write a record.

    Returns the parsed Pydantic object (same as llm.with_structured_output(schema).invoke(messages)).
    """
    import os as _os
    import time as _time
    _tel_dir = _os.environ.get("SCIML_TELEMETRY_DIR")

    if not _tel_dir:
        # No telemetry — call normally, with retry on None
        for _attempt in range(3):
            _result = llm.with_structured_output(schema).invoke(messages)
            if _result is not None:
                return _result
            if _attempt < 2:
                print(f"[retry] {agent_name} returned None, retrying in 10s...")
                _time.sleep(10)
        return _result  # return None after 3 failures, let caller handle

    # Telemetry enabled: use include_raw to capture usage metadata, with retry on None
    for _attempt in range(3):
        t0 = _time.time()
        raw_result = llm.with_structured_output(schema, include_raw=True).invoke(messages)
        latency = _time.time() - t0
        if raw_result.get("parsed") is not None:
            break
        if _attempt < 2:
            print(f"[retry] {agent_name} returned None, retrying in 10s...")
            _time.sleep(10)

    parsed = raw_result.get("parsed")
    raw_msg = raw_result.get("raw")

    # Extract tokens from raw AIMessage
    input_tokens = 0
    output_tokens = 0
    token_source = "unknown"

    if raw_msg is not None:
        usage = getattr(raw_msg, "usage_metadata", None)
        if usage:
            input_tokens = usage.get("input_tokens", 0) or 0
            output_tokens = usage.get("output_tokens", 0) or 0
            token_source = "usage_metadata"
        if token_source == "unknown":
            # Fallback: response_metadata (OpenAI / some providers)
            meta = getattr(raw_msg, "response_metadata", {}) or {}
            token_info = meta.get("token_usage", meta.get("usage", {}))
            if token_info:
                input_tokens = token_info.get("prompt_tokens", token_info.get("input_tokens", 0)) or 0
                output_tokens = token_info.get("completion_tokens", token_info.get("output_tokens", 0)) or 0
                token_source = "response_metadata"

    model_name = AGENT_MODELS.get(agent_name, "")
    from telemetry import compute_cost, LLMCallRecord, write_llm_record
    from datetime import datetime as _dt
    in_cost, out_cost, total_cost = compute_cost(model_name, input_tokens, output_tokens)
    total_tokens = input_tokens + output_tokens
    throughput = total_tokens / latency if latency > 0 else 0.0

    record = LLMCallRecord(
        timestamp=_dt.utcnow().isoformat(),
        agent_role=agent_name,
        model=model_name,
        iteration=int(_os.environ.get("SCIML_TELEMETRY_ITERATION", -1)),
        solution_id=_os.environ.get("SCIML_TELEMETRY_SOLUTION_ID", "unknown"),
        pid=_os.getpid(),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_cost_usd=in_cost,
        output_cost_usd=out_cost,
        total_cost_usd=total_cost,
        latency_seconds=round(latency, 3),
        throughput_tps=round(throughput, 1),
        token_source=token_source,
    )
    write_llm_record(record, _tel_dir)

    return parsed


# ============================================================================
# Prompt Templates and Stencils
# ============================================================================

EVALUATE_PY_STENCIL = '''
# <any_necessary_imports>

# This path is relative to the execution CWD (e.g., .../solution_00/)
CHECKPOINT_PATH = "./MODEL_CHECKPOINT"

# <any_additional_helper_functions>

def get_test_data():
    """
    Generated by Tester Agent based on user's evaluation.md.
    Usually you are supposed to do one of the following:
    1. Load data from a file in the solution directory (e.g., ./val_data.npz) as specified by the user
    2. Generate synthetic data according to user's specs, maybe multiple datasets, 
         e.g., sample from specific functions, grids, Gaussian random fields, or other distributions
         e.g., use tranditional numerical methods (FDM, FEM) to generate high-fidelity solutions as ground truth
    3. Pure physics-informed learning: sample collocation points in the domain as test data input, and the output is computed via the PDE residuals
    4. A combination of the above (e.g., some data-driven points + some collocation points) 
    """
    pass

def load_model_from_checkpoint(path = CHECKPOINT_PATH):
    """
    Loads the model based on the contract in guidelines.md.
    The model checkpoint is located at the given path CHECKPOINT_PATH.

    CRITICAL: You MUST import the MODEL class from solution.py, then load weights. 

    Example for PyTorch:
        from solution import MODEL
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MODEL()
        model.load_state_dict(torch.load(path, map_location=device))
        model = model.to(device)
        return model
        
    DO NOT use torch.load(path) directly without importing MODEL class first!
    """
    pass

def compute_success_metric(model, test_data) -> float:
    """
    Generated by Tester based on user's evaluation.md.
    Computes the success metric as exactly as defined by the user.
    You are supposed to:
    1. Use the model to make predictions on the test_data
    2. Compute the metric as defined by the user
        For example, if it's data-driven, compute MSE, relative l2, or any other user-defined metric against ground truth;
        if it's physics-informed, use automatic differentiation to compute PDE residuals, BC/IC errors, or any other user-defined metric.
    3. Print all intermediate metrics and observations to stdout
        For example, if you are asked to test on multiple datasets, you should print out the errors on each dataset along with description for what that dataset is.
        Or if you have multiple components in your final loss, such as PDE residual, BC error, IC error, you should print out the loss for each term
        Additionally, print out any other information to assess the model performance that you deem helpful. The more detailed the printing, the better.
    4. Return the final scalar score

    If you need post-processing (e.g., gradients): Call Engineer's API (NOT autograd)
    Example:
        # gradients = model.compute_gradient(x_test)
        # laplacian = model.compute_laplacian(x_test)
    """
    # Compute predictions
    # <your code here>

    # Compute metrics
    # errors = np.abs(predictions - ground_truth)
    score = 0.0 # Placeholder

    # Report distributions as TEXT (LLMs can't see plots)
    # Example:
    # print(f"Error distribution: min={errors.min():.6f}, "
    #       f"p25={np.percentile(errors, 25):.6f}, "
    #       f"median={np.median(errors):.6f}, "
    #       f"p75={np.percentile(errors, 75):.6f}, "
    #       f"max={errors.max():.6f}")

    # Visualization for human viewing (optional, try-except to avoid breaking evaluation)
    try:
        import matplotlib.pyplot as plt
        # Example: plot error distribution
        # plt.figure()
        # plt.hist(errors, bins=50)
        # plt.xlabel('Absolute Error')
        # plt.savefig('error_dist.png')
        # plt.close()
    except Exception as e:
        print(f"Warning: Could not save plot: {e}")

    return score

if __name__ == "__main__":

    print("--- EVALUATION START LOG ---")
    try:
        # All function print statements go to stdout and are captured.
        test_data = get_test_data()
        model = load_model_from_checkpoint(CHECKPOINT_PATH)
        score = compute_success_metric(model, test_data)

        # Report the final scalar score as a parseable JSON to stdout (required for main script)
        print("--- FINAL SCALAR METRIC ---")
        print(json.dumps({"status": "success", "score": score, "message": "Evaluation completed successfully."}))

    except Exception as e:
        # Report failure to stdout
        print(json.dumps({"status": "error", "message": str(e)}))
        sys.exit(1)         # Exit with a non-zero code to signal failure
'''

GUIDELINES_MD_STENCIL = '''
# Engineering Contract & Guidelines

You MUST adhere to the following rules for your code to be evaluated. Failure to do so will result in an integration error.

1. Your main model class/functions MUST be named MODEL.
   This class/function MUST implement the model architecture and forward pass/prediction logic.

2. Your model MUST accept input data of shape ... , dtype ... , and output data of shape ... , dtype ... .
   During evaluation, your model will be loaded and tested on private data of this shape and type.

3. You MUST implement model checkpointing as follows:
   - You MUST save your model's weights/state to the path `./MODEL_CHECKPOINT` when the script is run.
   - **IMPORTANT**: Save ONLY the weights/state_dict, NOT the entire model object
   - Framework-specific requirements:
     * PyTorch: Use `torch.save(model.state_dict(), './MODEL_CHECKPOINT')`
     * JAX/Equinox: Use `eqx.tree_serialise_leaves(path, model)`
     * TensorFlow: Use `model.save_weights('./MODEL_CHECKPOINT')`
   - The checkpoint will be loaded in evaluate.py by importing your MODEL class and loading weights

4. (If evaluation requires post-processing beyond direct inference):
   You MUST implement these methods in MODEL class:
   - ... (Tester specifies if needed, e.g., compute_gradient(x) → returns ∂u/∂x)
   - ... (shapes/types for each method)

5. ...

---
# Code Stencil (The Engineer MUST use this structure)
# This stencil should be modified according to the specific problem and sent to the engineer as part of the contract.
---

import argparse
import os
import sys
# ... all other necessary imports (e.g., jax, tensorflow, torch) ...

# The model class/functions *must* be named MODEL
# Implement as a class or set of functions
class MODEL:
    ... (Engineer's Implementation) ...
    ... (include all class API methods required by this guideline) ...

# ... (Any additional helper functions) ...

def main(mode):
    # Instantiate the model
    model = MODEL()

    if mode == "validate":
        # Run a minimal "smoke test"
        print("Running in VALIDATE mode (1 epochs)...")
        run_training_loop(model, epochs=1)

    elif mode == "train":
        # Run the full, expensive training
        print(f"Running in TRAIN mode ({NUM_EPOCHS} epochs)...")
        run_training_loop(model, epochs=NUM_EPOCHS)

    # Save checkpoint to the standard path
    model.save_state("./MODEL_CHECKPOINT")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["validate", "train"],
        help="Set execution mode: 'validate' (fast test) or 'train' (full run)"
    )
    args = parser.parse_args()
    main(args.mode)

During the testing phase, I'll test your model submission by running:
(... summarize here how `evaluate.py` works ...)
(... include the exact code snippets of model loading and testing, with clear comments on the shapes and types of data expected, but NEVER the data generation part or data itself ...)
(... NEVER include any suggestions for algorithms, model architectures, optimizers, etc. ...)
'''

TESTER_SYSTEM_PROMPT = """You are an expert software tester designing a testing contract for scientific machine learning problems.

Your role: You must make ALL technical decisions about testing, model interfaces, and data formats. The Engineer will follow your contract exactly, so your decisions must be:
1. **Consistent**: evaluate.py and guidelines.md must match perfectly
2. **Realistic**: Decisions must be implementable
3. **Unambiguous**: Every interface detail must be specified
4. **Complete**: No missing information that Engineer would need

Key responsibilities:
- Decide model checkpoint format and specify EXACT serialization/deserialization method
- Specify exact MODEL class interface (input/output shapes, dtypes, methods)
- Generate comprehensive test data (but NEVER leak actual data in guidelines)
- Design tricky test cases and edge cases to robustly evaluate the solution
- Ensure evaluate.py can load and test ANY solution following guidelines.md

**VERY IMPORTANT: DELEGATION PRINCIPLE:**
If evaluation requires operations beyond direct model inference, delegate to Engineer:
Example: If you need gradients during evaluation (e.g., PDE residual computation):
- DON'T: Use torch.autograd in evaluate.py (may not work with custom architectures)
- DO: Require Engineer to implement model.compute_gradient(x) in guidelines.md
- Your evaluation script calls: gradients = model.compute_gradient(x_test)
You should make the decision whether such methods are needed based on the problem. 
- If direct model inference produces all outputs required for scoring, do NOT ask Engineer to implement extra methods.
- If extra computations (gradients, laplacians, etc.) on top of model prediction are needed for scoring, you MUST require Engineer to implement these methods.
For any methods you ask the Engineer to implement, you MUST specify:
    * Exact method name (but NOT how the method should be implemented)
    * Input shapes and dtypes
    * Output shapes and dtypes
    * Any other relevant details or constraints
You MUST NOT specify HOW the method should be implemented - that is up to the Engineer. You should specify only WHAT the method should do (input/output contract).
- DON'T: "compute gradients using autograd"
- DO: "implement compute_gradient(x) that returns ∂u/∂x with input x of shape (...) and output of shape (...)"
Same principle applies for any post-processing operations.

**Framework-Specific Checkpointing:**
**PyTorch**: You MUST specify state_dict() saving, NOT whole model saving:
  ✓ CORRECT in guidelines.md:
    "Save checkpoint: torch.save(model.state_dict(), './MODEL_CHECKPOINT')"
  ✗ WRONG:
    "Save checkpoint: torch.save(model, './MODEL_CHECKPOINT')"

  ✓ CORRECT in evaluate.py:
    from solution import MODEL
    model = MODEL()
    model.load_state_dict(torch.load(checkpoint_path))
  ✗ WRONG:
    model = torch.load(checkpoint_path)  # Fails when MODEL class not in evaluate.py
**JAX/Equinox**: Use eqx.tree_serialise_leaves() and eqx.tree_deserialise_leaves()
**TensorFlow**: Use model.save_weights() and model.load_weights()

CRITICAL CONSTRAINTS - You are a SOFTWARE TESTER, NOT an ML architect:
- DO NOT give assumptions or hints about how to solve the problem
- DO NOT recommend model architectures (e.g., "use MLP", "use CNN", "use ResNet")
- DO NOT suggest activation functions (e.g., "recommend tanh", "use ReLU", "prefer smooth activations")
- DO NOT provide optimization strategies (e.g., "use Adam", "you are free to choose learning rate scheduling")
- DO NOT suggest algorithmic approaches (e.g., "use domain decomposition", "try ensemble methods")
- DO NOT ask model to be flexible to different data shapes/types - your data shapes and types are fixed and strictly followed
- DO NOT give any details about the validation data set (e.g., length scales, GRF parameters, sampling methods, domain regions, number of test points). Instead, just say "case 1, case 2, etc." without specifics.

WHAT YOU SHOULD SPECIFY:
✓ Exact input/output shapes and dtypes 
✓ Model class definition and interface
✓ Checkpoint name ('MODEL_CHECKPOINT'), format and serialization method
✓ Success metrics

IMPORTANT - for models with multiple inputs/outputs, specify shapes/types for each tensor clearly.
For example: 
```
Your DeepONet must take branch input of shape (N, 128) and trunk input of shape (N, 2), 
where N is batch size, 128 is the branch function discretization size, and 2 is the spatial dimension (x,y), 
and output predictions of shape (N, 128), where 128 corresponds to the output function discretization size.
```

Let the engineer know that it may choose not to use some of the inputs if desired, but the interface must be followed exactly.
For example: 
```
If your model can predict the output using only the first input, that is acceptable,
but the input interface must still accept both first and second inputs as specified,
so that when evaluate.py calls your model, it works without error.
```

Let the engineer know that it may choose not to make a machine learning model at all if desired (e.g., purely analytical solution), but the interface must still be followed exactly.
For example: 
```
If you choose to implement an analytical transformation of data A to data B without any machien learning, that is acceptable,
but you must still implement the MODEL class with the specified input/output interface: model should take in input A of shape (...) and output B of shape (...), 
and when the evaluate.py calls your model, it can produce the expected outputs from the corresponding inputs without error.
```

Visualization Requirements:
Your evaluate.py should include visualization to help understand model performance:
- Save plots to the current working directory (solution directory)
- Create visualizations appropriate for the problem type, the more comprehensive the better. For example:
  * Spatial PDEs: Plot solution heatmaps, residual distributions across domain
  * Data-driven: Plot prediction vs ground truth, error distributions
- Wrap visualization code in try-except to ensure it NEVER breaks evaluation

TEST DESIGN GUIDELINES: 
You are free to design your own testing data, and you should consider to 
- Design diverse test cases that robustly evaluate the model. For example: 
    - Boundary and initial conditions at extremes of domain
    - Discontinuities, sharp gradients, or multi-scale features
    - Any other challenging but realistic scenarios relevant to the problem
- For each case of your own design, you must print out a description for this case along with its results
- Must make sure the ground truth is accurate and high-fidelity. 

**CRITICAL - Detailed Diagnostic Printing:**
Your compute_success_metric() function MUST print detailed diagnostic information to stdout:

✓ GOOD EXAMPLES:
- "BC Error on left boundary: 0.0023"
- "BC Error on right boundary: 0.0045"
- "BC Error on top boundary: 0.0012"
- "PDE residual in subdomain 1 (x<0): 0.0034"
- "PDE residual in subdomain 2 (x>0): 0.0056"
- "Relative L2 error: 2.3%"
- "MSE: 0.00045"
- "Max absolute error: 0.012"
- "Interior point residual MSE: 0.0023"
- "Boundary condition violation MSE: 0.0045"

✗ BAD EXAMPLES (too vague):
- "Error: 0.005" (which error? where?)
- "Loss: 0.1" (what type of loss? on what data?)

Print breakdowns whenever meaningful:
- If domain has multiple boundaries → print error on EACH boundary separately
- If domain is separable into subdomains → print metrics for EACH subdomain
- If multiple loss components (PDE residual, BC error, IC error) → print EACH component
- Include multiple error metrics (MSE, relative L2, max absolute error) when applicable
- Include any other insightful observations about model performance that you can think of. Must help analysts understand where the model succeeds or fails.

Remember: ONLY specify the testing interface and contract. Private test data stays in evaluate.py only. Guidelines describe data shapes/types, NOT actual values or implementation strategies.

**CRITICAL - Evaluation Script Requirements:**

Your evaluate.py MUST:
1. Compute extensive and objective metrics (MSE, relative L2, max error, etc.)
2. Return a single scalar final SCORE (lower is better) (for example, a sum of all error terms)
3. NEVER implement pass/fail logic based on thresholds (for example, "Success Criteria Check: MSE < 0.01" should NEVER be allowed). You should NOT decide on any pass/fail threshold. You should ONLY produce the metrics and score.
4. Include the MANDATORY JSON output format (see below)

**Testing Data Privacy:**
You MUST NOT leak any information about the testing data into guidelines.md.

NEVER include in guidelines.md:
- Number of test points
- Sampling strategy
- Data generation method
- Domain regions
- Specific test cases or scenarios

ONLY include in guidelines.md:
- Data shapes/types (e.g., "input: (N, 2), output: (N, 1)")
- General description (e.g., "spatial points in domain")

**JSON Output Format (REQUIRED - Infrastructure Code):**
This MUST be included VERBATIM in your evaluate.py:

```python
if __name__ == "__main__":
    print("--- EVALUATION START LOG ---")
    try:
        test_data = get_test_data()
        model = load_model_from_checkpoint(CHECKPOINT_PATH)
        score = compute_success_metric(model, test_data)

        # MANDATORY: JSON output for system parsing
        print("--- FINAL SCALAR METRIC ---")
        print(json.dumps({"status": "success", "score": score, "message": "Evaluation completed successfully."}))

    except Exception as e:
        # Only use "error" status for exceptions (file not found, import errors)
        print(json.dumps({"status": "error", "message": str(e)}))
        sys.exit(1)
```

**Key Points:**
- "status": "success" means the evaluation RAN successfully (code executed without errors)
- "status": "error" means there was an EXCEPTION (file missing, import failed, runtime error)
- DO NOT use "status": "error" because model performance is poor
- The "score" field is a single scalar float representing overall performance (lower is better)
- This JSON format MUST be included VERBATIM - it's infrastructure code
"""

ENGINEER_SYSTEM_PROMPT = """You are an expert ML engineer implementing scientific machine learning solutions.

Your job: Generate complete, working solution code that follows the engineering contract EXACTLY.

Key responsibilities:
- Read problem description, requirements, and guidelines carefully
- Implement MODEL class following the exact interface specified in guidelines
- Implement checkpoint save/load as specified
- Create validate mode (fast, 1 epoch) and train mode (full training)
- Print training progress to stdout

CRITICAL CONSTRAINTS:
- Follow guidelines.md specifications EXACTLY (shapes, dtypes, checkpoint format)
- Implement both --mode=validate and --mode=train
- Save checkpoint to ./MODEL_CHECKPOINT
- Self-contained code (no external scripts needed)
- Print comprehensive training logs
- Must follow any additional proposal or guidelines provided. Implement only what is specified and all that is specified.

**CRITICAL - Checkpoint Saving (Training May Timeout)**:
1. **If there's model training:** Save the BEST model immediately whenever it improves during training
   - Track best metric, overwrite ./MODEL_CHECKPOINT each time model improves
   - Save frequently enough to avoid losing progress if training is interrupted

**CRITICAL - GPU Usage**:
Your code MUST automatically detect and use available hardware:

For PyTorch:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# Move all tensors to device during training:
x_train = x_train.to(device)

For JAX:
# JAX auto-selects GPU by default, no explicit device handling needed

For TensorFlow:
# TensorFlow auto-detects GPU, or explicitly:
gpus = tf.config.list_physical_devices('GPU')

The execution environment will assign GPUs via environment variables.
Do NOT hardcode specific GPU IDs (e.g., cuda:0, cuda:1) in your code.

2. **Training Visualization**:
Add visualization code during training to monitor progress:
- Save plots to current working directory (will be solution_XX/)
- Examples: loss curves, prediction samples at checkpoints
- Use try-except to ensure visualization NEVER breaks training
- Only create plots that are meaningful for this specific problem
- Do NOT save plots every epoch to avoid excessive I/O; save every N epochs or at key milestones only

Example pattern:
try:
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('loss_curve.png')
    plt.close()
except Exception as e:
    print(f"Warning: Could not save plot: {e}")

3. **Non-training Problems (very rare, unless explicitly specified):**
If the problem does NOT involve model training (e.g., purely analytical solutions, data processing):
- You should still implement both --mode=validate and --mode=train
- You should still implement checkpoint saving to ./MODEL_CHECKPOINT and the MODEL class
- In the MODEL class, you can do whatever is appropriate for the problem, even if it doesn't involve training
- Your MODEL class call will produce the expected outputs as per guidelines.md, even if no training occurs

4. **Framework Translation:**
If any reference code uses a different framework than required, you MUST translate the implementation to the required framework:
- Study the reference implementation carefully to understand the algorithm
- Implement the SAME algorithm using the required framework's idioms
- Preserve the mathematical operations and logic
- Adapt framework-specific features
Use the reference code as a guide for WHAT to implement, not HOW to write it in your target framework.

5. **Some helpful tips to review:**
- Use torch.autograd.grad() instead of .backward() for cleaner graph management. Properly detach intermediate gradients after taking automatic differentiation to avoid memory leaks.
- Ensure all tensors are on the correct device (CPU/GPU) during operations.
- Ensure the tensor shapes match exactly as specified in guidelines.md and evaluate.py.

**Important: MODEL class methods and APIs:**
- You might be asked to implement additional methods in MODEL class (e.g., compute_gradient(x)) by the tester's guidelines/contract
- Implement these methods EXACTLY as specified, with correct input/output shapes and types, calculating the required quantities

Your code will be executed as-is. Make it work on the first try."""

VALIDATOR_SYSTEM_PROMPT = """You are a debugging expert analyzing validation failures.

Your job: Determine who is responsible for the validation error.

Given:
- guidelines.md (testing contract)
- solution.py (engineer's implementation)
- evaluate.py (test script)
- Error traceback

**IMPORTANT:** Validation failure means the code didn't RUN successfully, not that
the model performance was poor. A successful validation means:
- solution.py executed without errors
- evaluate.py executed without errors
- A score was produced

Determine root cause:

"tester_error" if TESTER is at fault:
- evaluate.py crashes or has runtime errors
- evaluate.py doesn't produce required JSON output
- guidelines.md is ambiguous or incorrect
- calculations done in evaluate.py are wrong: 
    - Failed to convert model tensor output to numpy correctly
    - Incorrectly use torch tensor functions on float objects, such as torch.mean(error) when error is a float
    - Incorrectly used torch.no_grad() or torch.inference_mode() around gradient computations
- Examples: "evaluate.py missing JSON output", "guidelines specify wrong data shape"
- CRITICAL: evaluate.py uses torch.no_grad() or torch.inference_mode() around gradient computations
  (see detailed gradient diagnostics below)

"engineer_error" if ENGINEER is at fault:
- solution.py crashes or has runtime errors
- solution.py doesn't follow the interface in guidelines.md
- MODEL class is not implemented as specified by guidelines.md

**CRITICAL - Gradient Computation Error Diagnostics:**

When you see errors like:
- "element 0 of tensors does not require grad and does not have a grad_fn"
- "RuntimeError: grad can be implicitly created only for scalar outputs"
- "One of the differentiated Tensors does not require grad"

Follow this diagnostic checklist IN ORDER:

1. **First, check evaluate.py for gradient-disabling contexts** (MOST COMMON):
   - Look for: `with torch.no_grad():`, `@torch.no_grad()`, `with torch.inference_mode():`
   - If ANY of these wrap calls to autograd.grad(), PDE residual computation, or derivative calculation → **TESTER ERROR**
   - torch.no_grad() COMPLETELY disables gradient tracking regardless of any other settings
   - Even if model is in train() mode and tensors have requires_grad=True, torch.no_grad() will break autograd

2. **Check if evaluate.py creates tensors without requires_grad=True**:
   - For PDE/physics-informed problems, input tensors MUST have requires_grad=True
   - If evaluate.py creates tensors like: `x = torch.tensor(data)` without `requires_grad=True` → **TESTER ERROR**
   - Correct: `x = torch.tensor(data, requires_grad=True)`

3. **Check if evaluate.py uses create_graph=True for higher-order derivatives**:
   - For computing Laplacians, Hessians, or any second+ derivatives
   - First derivative: `grad(..., create_graph=True)` is REQUIRED
   - If missing → **TESTER ERROR**

4. **Only if above checks pass, check solution.py**:
   - MODEL class produces wrong output shapes/types
   - MODEL class has broken forward pass
   - This is rare for gradient errors → **ENGINEER ERROR**

**IMPORTANT CLARIFICATIONS:**
- model.eval() mode does NOT disable gradients! It only affects dropout/batchnorm behavior
- Do NOT suggest fixing model.eval() for gradient errors - this shows misunderstanding
- The issue is almost always torch.no_grad() in evaluate.py, NOT model.eval() in solution.py

**Example correct diagnosis:**
Error: "does not require grad and does not have a grad_fn"
Evaluate.py has: `with torch.no_grad(): residual = compute_pde_residual(...)`
→ Diagnosis: **TESTER ERROR** - "evaluate.py wraps compute_pde_residual() in torch.no_grad() context (line X). This disables gradient tracking needed for automatic differentiation. Remove the torch.no_grad() wrapper from this computation."

**DO NOT mark as error if:**
- Model performance is poor but code runs fine
- Score is high but evaluation completed successfully
- Model doesn't meet any performance targets but produces valid output

Analyze carefully and return structured decision with specific feedback."""


DEBUGGER_SYSTEM_PROMPT = """You are an experienced debugging assistant for scientific ML code. Your ONLY job is to help fix implementation errors.

**CRITICAL CONSTRAINTS - What You Should NOT Suggest:**
1. ❌ Strategic decisions (e.g., "Use a deeper network", "Add more layers")
2. ❌ Algorithms or research ideas (e.g., "Try adaptive activation functions", "Implement attention mechanism")
3. ❌ Problem-solving approaches (e.g., "Implement physics-informed loss", "Use domain decomposition")
4. ❌ What the solution should do differently at a high level
5. ❌ Install new libraries or dependencies (e.g., "Install tensorflow-datasets", "Install DeepXDE"), because the environment is fixed. Find a workaround using existing libraries.

**ONLY Provide:**
✅ Implementation fixes:
   - Shape mismatches: "Tensor X has shape (10, 2) but function expects (10, 1). Use reshape() or transpose()"
   - Missing imports: "Add 'import torch.nn as nn' at the top"
   - Syntax errors: "Indentation error on line 45 - code block not aligned"
   - Type errors: "Variable 'loss' is None, ensure it's initialized before use"
   - Contract violations: "Checkpoint not saved - add torch.save(model.state_dict(), './MODEL_CHECKPOINT')"

**Examples of GOOD vs BAD suggestions:**
❌ BAD: "The model is underfitting. Try increasing the number of layers and using a larger hidden dimension."
✅ GOOD: "RuntimeError: Expected tensor of size [100, 64] but got [100, 32]. Check the hidden_dim parameter matches between layers."

❌ BAD: "The loss isn't decreasing well. Consider using adaptive learning rate schedules or curriculum learning."
✅ GOOD: "NameError: 'optimizer' is not defined. Move optimizer initialization before the training loop (line 45)."

Your job is to analyze errors and provide SINGLE, CONCISE, ACTIONABLE debugging suggestions (2-4 sentences maximum)."""


RETRIEVER_SYSTEM_PROMPT = """You are a critical knowledge base retriever for scientific ML research.

**Your Role:**
- Critically evaluate KB entries for relevance and feasibility
- Check if techniques are already implemented in current solution
- Connect problem characteristics with KB solutions
- Be highly selective - return None if no entry truly helps
- Focus on what will improve performance based on analysis

**Mindset:**
You are CRITICAL and SELECTIVE, not eager to suggest. Most of the time, no KB entry is needed. Only select an entry if it will make a significant, feasible improvement."""


ANALYST_SYSTEM_PROMPT = """You are an experienced, critical ML research analyst.

**Your Role:**
- Provide objective, factual analysis based ONLY on observed data
- Be concise and results-focused
- Identify problems but do NOT suggest solutions
- Focus on algorithmic/mathematical issues, not hardware limitations
- Write descriptive, self-contained reports for future reference
- If plots/visualizations are provided, analyze them to extract insights about training dynamics and performance

**Mindset:**
- Critical - expose problems clearly without sugar-coating
- Experienced - understand ML training dynamics deeply
- Objective - no speculation, only evidence-based conclusions
- Factual - quantify observations with metrics from logs
- Concise - prioritize signal over noise, keep analysis actionable

**Plot Analysis (if plots are provided):**
- Describe what the visualizations show clearly and concisely
- Extract key patterns, trends, or anomalies visible in plots
- Connect visual observations to performance and training dynamics
- Remember: downstream agents cannot see the plots, so describe them as if explaining to someone who is blind"""


PROPOSER_SYSTEM_PROMPT = """You are an experienced scientific ML research analyst and proposal generator.

**Your role varies by round:**
- **Early rounds (reasoning)**: Conduct deep mathematical and theoretical analysis
- **Synthesis round**: Form concrete implementation plan from reasoning
- **Final round**: Make definitive, implementation-ready proposal

**Key traits:**
- Mathematical - analyze problems at fundamental level rigorously
- Numerical - use mathematical insights to guide numerical algorithms
- Think step by step - must show detailed reasoning and derivations
- Innovative - propose novel, cutting-edge approaches grounded in reasoning
- Dialogic - engage constructively with critic's feedback
- Decisive - make clear, actionable recommendations

**Remember:**
- Theoretical depth before implementation details
- Think creatively within scientific ML constraints
- Balance innovation with feasibility
- Strongly consider using Knowledge Base solutions when relevant
- Do NOT be afraid to propose complicated but sound model architecture or implementation ideas
- Specify exact hyperparameter values, never vague suggestions"""


CRITIC_SYSTEM_PROMPT = """You are a critical evaluator of scientific ML reasoning and plans. You help a proposer improve their proposals through rigorous critique.

**Your role varies by round:**
- **Early rounds**: Critique mathematical and theoretical reasoning soundness
- **Plan round**: Critique implementation feasibility and specificity

**Key traits:**
- Rigorous - demand mathematical soundness
- Constructive - challenge to improve, not to obstruct
- Fair - recognize sound arguments, admit when proposer is right
- Dialogic - engage in genuine intellectual exchange
- Goal oriented - the goal is to arrive at the best possible solution

**Remember:**
- Ask probing questions to deepen analysis
- Do NOT overwhelm proposer with many new ideas at once. The end goal is to find ONE best proposal, not to brainstorm many different ideas
- Focus on the proposer's ideas
- Encourage proposer to use Knowledge Base solutions when relevant and helpful
- Do NOT discourage complicated but sound model architecture or implementation ideas
"""


# ============================================================================
# Agent Functions
# ============================================================================

def tester_agent(problem: str, requirements: str, evaluation: str,
                 dataset_info: str = "", refinement_feedback: str = "") -> ContractOutput:
    """
    Tester agent generates or refines evaluate.py and guidelines.md

    Args:
        problem: Problem description from problem.md
        requirements: Requirements from requirements.md
        evaluation: Evaluation strategy from evaluation.md
        dataset_info: Optional dataset information from dataset_config.json
        refinement_feedback: Optional feedback for refinement (empty for initial generation)

    Returns:
        ContractOutput with evaluate_py and guidelines_md
    """
    llm = get_llm("tester")

    if not refinement_feedback:
        # Initial generation
        dataset_section = ""
        if dataset_info:
            dataset_section = f"""
**Evaluation Dataset Information:**
{dataset_info}

The user has provided this above mentioned evaluation dataset. 
You MUST use this in your evaluate.py for testing the final model based on the user's Evaluation Metrics criteria.
  - Load validation data in get_test_data() based on user provided loading instructions
  - Perform any operation that the user requested on this validation set
  - The Engineer will NOT see validation data during training
  - Document the testing interface using the dataset and evaluation metrics in guidelines accordingly
Example 1: if the user provided a dataset of spatial points and solution values, you should specify in guidelines that the model will be tested on input of spatial points of shape (...) and output is solution values of shape (...).
Example 2: if the user provided a dataset of an observable (e.g., sensor readings), you should specify in guidelines how the model APIs should generate the predictions for this observable (e.g., input shape, output shape).
"""

        prompt = f"""Generate a complete testing contract for the following problem:

**Problem Description:**
{problem}

**Implementation Requirements:**
{requirements}

**Evaluation Metrics:**
{evaluation}
{dataset_section}

You must generate TWO files:

1. **evaluate.py**: A complete, self-contained Python test script following this stencil:
{EVALUATE_PY_STENCIL}

2. **guidelines.md**: Engineering contract following this stencil (you should fill in placeholders without defining any exact model architecture or functions):
{GUIDELINES_MD_STENCIL}

**Critical Instructions:**
- Make ALL technical decisions (checkpoint format, data shapes)
- If evaluation requires operations beyond direct model inference, delegate to Engineer to implement class methods. 
- In guidelines.md, describe ONLY shapes/types, NOT actual test data
- Ensure evaluate.py and guidelines.md are 100% consistent
- Eliminate ALL ambiguities - be specific about every detail 
- Do NOT suggest or write code for any of the following: model architectures, activation functions, optimization strategies, algorithms, or any other problem-solving approaches
- Do NOT analyze the problem (for example: "this is a stiff PDE, so use implicit methods" is NOT allowed
- Remember: Do NOT explain HOW to implement - specify WHAT to implement.
- DO NOT give any details about the validation data set (e.g., length scales, GRF parameters, sampling methods, domain regions, number of test points). Instead, just say "case 1, case 2, etc." without specifics.

**You MUST follow these when generating guidelines.md:**
- This is a contract for the engineer to follow exactly, not for you to explain your decisions or how evaluation is done. 
- Only include information that the engineer needs to implement the MODEL class and checkpointing
- The engineer is NOT in charge of evaluating its model - Do NOT ask it to implement any evaluation logic 
- Do NOT give engineer any information about validation dataset (including loading instructions) beyond shapes/types

Generate the complete code for both files now."""
    else:
        # Refinement based on feedback
        dataset_section = ""
        if dataset_info:
            dataset_section = f"""
**Dataset Information:**
{dataset_info}
"""

        prompt = f"""Refine the testing contract based on the feedback.

**Original Problem:**
{problem}

**Requirements:**
{requirements}

**Evaluation:**
{evaluation}
{dataset_section}

**Feedback:**
{refinement_feedback}

The feedback from running your code has found issues with the testing contract. Fix these issues and generate UPDATED versions of both evaluate.py and guidelines.md.

Ensure consistency between the two files and address all feedback points."""

    messages = [
        SystemMessage(content=TESTER_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]

    result = _llm_invoke(llm, ContractOutput, messages, "tester")
    return result


def engineer_agent(problem: str, requirements: str, guidelines: str,
                   training_set_info: str = "", refinement_feedback: str = "",
                   champion_code: str = "", proposal: str = "",
                   current_solution_code: str = "", agent_name: str = "engineer") -> SolutionOutput:
    """
    Engineer agent generates or refines solution.py

    Args:
        problem: Problem description from problem.md
        requirements: Requirements from requirements.md
        guidelines: Engineering guidelines from guidelines.md
        training_set_info: Training dataset information (filename, description, loading instructions)
        refinement_feedback: Optional feedback for refinement (debugging)
        champion_code: Parent solution code being mutated (named champion_code for backwards compatibility)
        proposal: Optional proposal markdown (for Phase 3 children)
        current_solution_code: Optional current solution code (for debugging iterations)
        agent_name: Agent role name for LLM selection (default: "engineer")

    Returns:
        SolutionOutput with solution_code
    """
    llm = get_llm(agent_name)

    # Build training data section if provided
    training_section = ""
    if training_set_info:
        training_section = f"""
**Training Dataset:**
{training_set_info}

Use this training data during your solution development. The data files will be available in your solution directory.
You MUST
- Load the training data as per the provided loading instructions exactly. Otherwise, the solution will fail. 
- Use this training data to train the model.
- Feel free to augment, denoise, or preprocess the data as needed, but the initial loading MUST follow the provided instructions exactly and this dataset MUST be used.
"""

    # Phase 3: Child solution generation (has parent and proposal)
    if champion_code and proposal and not refinement_feedback:
        prompt = f"""Generate solution code by implementing the following proposal on top of the parent's code.

**Problem:**
{problem}

**Requirements:**
{requirements}

**Engineering Guidelines (MUST FOLLOW EXACTLY):**
{guidelines}

{training_section}

**Parent's Code (reference from mutation source - use as foundation):**
```python
{champion_code}
```

**Proposal (implementation details to follow exactly - MUST respect):**
{proposal}

Generate COMPLETE solution.py that:
1. Implements the proposal's changes on top of the parent's approach
2. Follows all engineering guidelines exactly
3. Implements both --mode=validate (1 epoch) and --mode=train (full)
4. Saves checkpoint to ./MODEL_CHECKPOINT as specified
5. Prints training progress to stdout
6. Includes all hyperparameters and implementation details from the proposal

Generate the complete solution code now."""

    # Phase 2: Initial root solution (no champion or proposal)
    elif not refinement_feedback:
        prompt = f"""Generate complete solution code for the following problem.

**Problem:**
{problem}

**Requirements:**
{requirements}

**Engineering Guidelines (MUST FOLLOW EXACTLY):**
{guidelines}

{training_section}

Generate self-contained solution.py that:
1. Implements MODEL class following guidelines exactly
2. Implements both --mode=validate (1 epoch) and --mode=train (full)
3. Saves checkpoint to ./MODEL_CHECKPOINT as specified
4. Prints training progress to stdout

Generate the complete solution code now."""

    # Refinement based on debugging feedback
    else:
        prompt = f"""Fix solution code based on debugging feedback.

**Problem:**
{problem}

**Requirements:**
{requirements}

**Guidelines (MUST FOLLOW EXACTLY):**
{guidelines}

{training_section}

**Original Proposal (for context):**
{proposal if proposal else "No proposal (this is root solution debugging)"}

**Your Current Solution Code (that has bugs):**
```python
{current_solution_code}
```

**Debugging Feedback:**
{refinement_feedback}

Fix the issues and generate UPDATED solution.py that addresses all feedback while still following the original proposal (if applicable)."""

    messages = [
        SystemMessage(content=ENGINEER_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]

    result = _llm_invoke(llm, SolutionOutput, messages, agent_name)
    return result


def validator_agent(guidelines: str, solution_code: str, evaluate_code: str,
                   error_traceback: str) -> ValidationDecision:
    """
    Validator analyzes validation errors and determines culprit

    Args:
        guidelines: Content of guidelines.md
        solution_code: Content of solution.py
        evaluate_code: Content of evaluate.py
        error_traceback: Error message and traceback from validation run

    Returns:
        ValidationDecision with culprit and specific feedback
    """
    llm = get_llm("tester")  # Use same model as tester for analysis

    prompt = f"""Analyze this validation failure and determine root cause.

**Guidelines (testing contract):**
```
{guidelines}
```

**Solution Code:**
```python
{solution_code}
```

**Evaluation Code:**
```python
{evaluate_code}
```

**Error:**
```
{error_traceback}
```

Determine if the TESTER (evaluate.py or guidelines.md) or ENGINEER (solution.py) is responsible.
Provide specific feedback on what to fix."""

    messages = [
        SystemMessage(content=VALIDATOR_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]

    result = _llm_invoke(llm, ValidationDecision, messages, "validator")
    return result


def retriever_agent(champion_code: str, champion_analysis: str,
                   problem_description: str, kb_indices: list) -> KBRetrievalResult:
    """
    KB retriever agent to find 0-1 relevant knowledge entries.

    Args:
        champion_code: Current champion solution code
        champion_analysis: Analysis report of champion solution
        problem_description: User's problem description
        kb_indices: List of KB entry metadata from indices.json

    Returns:
        KBRetrievalResult with selected entry index and reasoning
    """
    llm = get_llm("retriever")

    # Format KB entries for the prompt
    kb_list = []
    for idx, entry in enumerate(kb_indices):
        kb_list.append(f"""
[{idx}] {entry['method_name']}
Description: {entry['description']}
""")

    kb_entries_text = "\n".join(kb_list)

    prompt = f"""**Your Task:** Review the available knowledge base entries and select 0-1 entry that will HELP IMPROVE the champion solution's performance and is FEASIBLE to implement.

**Problem Description:**
{problem_description}

**Champion Solution Analysis:**
{champion_analysis}

**Champion Solution Code:**
```python
{champion_code}
```

**Available Knowledge Base Entries:**
{kb_entries_text}

**Instructions:**
1. Critically evaluate each knowledge base entry for relevance
2. Check if the technique is already implemented in the champion's code (if so and it is implemented correctly, skip it - do not select it again since it won't help)
3. Consider the testing results - what kind of improvement would help? (e.g., if underfitting, avoid regularization techniques)
4. Determine if implementation is feasible on top of the current code
5. Return the index of the MOST HELPFUL entry, or None if no entry is relevant

**Important:**
- Return AT MOST 1 entry (0 or 1)
- Do NOT retrieve entries that are already implemented, UNLESS the implementation failed to follow the entry's method correctly or is showing completely different performance characteristics than what is shown in the entry. You can always retrieve something else. 
- Do NOT retrieve entries that are completely irrelevant to the problem
- Focus on what will improve performance based on the testing results
- The entry should be feasible to implement (not require a complete rewrite of unrelated components)

Provide detailed reasoning for your decision."""

    messages = [
        SystemMessage(content=RETRIEVER_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]

    result = _llm_invoke(llm, KBRetrievalResult, messages, "retriever")
    return result


def analyst_agent(solution_code: str, train_log: str, test_log: str,
                  score: float, proposal: str | None = None,
                  parent_analysis: str | None = None,
                  plot_images: list[tuple[str, str]] | None = None) -> str:
    """
    Analyst agent generates comprehensive performance analysis.

    Args:
        solution_code: Complete solution.py code (NO TRUNCATION)
        train_log: Complete training output from train_log.txt (NO TRUNCATION)
        test_log: Complete testing output from test_log.txt (NO TRUNCATION)
        score: Final evaluation score
        proposal: Proposal markdown (None for root solution) (NO TRUNCATION)
        parent_analysis: Parent's analysis (None for root solution) (NO TRUNCATION)
        plot_images: List of (filename, base64_encoded_image) tuples for plots (None if no plots)

    Returns:
        Markdown-formatted analysis report
    """
    # Prepare proposal section
    if proposal:
        proposal_section = "**Proposal** (what was attempted):\n" + proposal + "\n"
    else:
        proposal_section = "**Note**: This is the root solution (no proposal)."

    # Prepare parent analysis section
    if parent_analysis:
        parent_analysis_section = "**Parent Analysis** (for comparison):\n" + parent_analysis + "\n"
    else:
        parent_analysis_section = ""

    # Prepare parent comparison section for report structure
    if parent_analysis:
        parent_comparison_section = """
## Comparison with Parent
- What changed from parent solution? 
- Why the change helped or hurt performance?
- Lessons learned?"""
    else:
        parent_comparison_section = ""

    # Prepare plot analysis section for report structure (if plots provided)
    if plot_images and len(plot_images) > 0:
        plot_analysis_section = """

## Plot Analysis 
- Describe what the plots show (training curves, loss evolution, error distributions, etc.)
- Key observations and patterns visible in the visualizations (e.g., model prediction deviates from true solution in certain regions)
- How the plots relate to performance and training dynamics (e,g, oscillations in loss curves, etc.)
- Any other concerning trends or anomalies visible in plots"""
    else:
        plot_analysis_section = ""

    # Build text prompt
    prompt_text = f"""Analyze this solution's performance and generate a comprehensive report.

**Solution Code:**
```python
{solution_code}
```

**Training Log:**
```
{train_log}
```

**Testing Log:**
```
{test_log}
```

**Final Score**: {score} (lower is better)

{proposal_section}

{parent_analysis_section}

Generate a comprehensive analysis report in markdown format with these sections:

## Summary of Approach (2-3 sentences, be concise and clear)
- Brief description of the model architecture or algorithm used
- Key techniques or innovations implemented, focus on what is novel, non-traditional, and non-trivial
- Hyperparameters (network size, activations, weight for the loss terms etc.)
- Training setup (epochs, batch size, etc.)
- Optimization details (first and second order optimizer, learning rate, scheduler, etc.)

## Training Dynamics (if there's any training; skip if the problem is purely analytical and no model training occurred)
- Convergence: fast/slow/stagnant
- Loss components (if multiple): which dominates
- Training stability: stable/oscillating/diverging
- Overfitting/underfitting behavior
- Any numerical issues or anomalies observed

## Performance Breakdown
- Final score: {score}
- Component metrics (for examples, if available, different PDE residuals, BC errors, etc.) from test log
{"- Comparison with parent: improvements/regressions" if parent_analysis else ""}
- Identify algorithmic/mathematical bottlenecks from the logs (e.g., which loss term dominates, where errors concentrate)
- Do NOT comment on hardware bottlenecks (CPU/GPU usage, memory, etc.)

## Problems Identified (MUST be based on the observed training/testing results - NO speculation)
- Spot any potential algorithmic problems, numerical issues, or implementation flaws (only those that really stand out, not those that are trivial or non-essential)
- Examples: underfitting patterns, poor loss balance, insufficient expressivity, numerical instability
- EXPOSE problems only - do NOT propose solutions or suggest what to do next
- Focus on ALGORITHMIC problems, NOT hardware/resource problems

{parent_comparison_section}

{plot_analysis_section}

CRITICAL:
- If someone reads your report, they MUST be able to clearly understand what this method did and what happened to it during training and testing
- It must be self-contained and understandable, serving as a log for future reference
- Do NOT suggest code changes or improvements.
- Focus on understanding what happened and what problems exist both qualitatively and quantitatively, not what should be done next."""

    # Check if multimodal (with images) or text-only
    num_images = 0
    if plot_images and len(plot_images) > 0:
        # Use multimodal approach with images
        model_name = AGENT_MODELS["analyst"]
        temperature = TEMPERATURES["analyst"]
        num_images = len(plot_images)

        # Create Gemini LLM for multimodal support
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_output_tokens=60000
        )

        # Build message with images
        message_parts: list[MessageLikeRepresentation] = [prompt_text]

        # Add images
        for filename, image_b64 in plot_images:
            message_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"}
            })

        messages = [
            SystemMessage(content=ANALYST_SYSTEM_PROMPT),
            HumanMessage(content=message_parts)
        ]
    else:
        # Text-only approach
        llm = get_llm("analyst")
        messages = [
            SystemMessage(content=ANALYST_SYSTEM_PROMPT),
            HumanMessage(content=prompt_text)
        ]

    # Get structured output
    result = _llm_invoke(llm, AnalysisReport, messages, "analyst")

    # Combine analysis_markdown and plot_analysis if present
    final_analysis = result.analysis_markdown
    if result.plot_analysis:
        # Insert plot analysis section into the markdown
        final_analysis += f"\n\n## Plot Analysis\n\n{result.plot_analysis}"

    return final_analysis


def debugger_agent(
    error_message: str,
    error_traceback: str,
    solution_code: str,
    problem: str,
    requirements: str,
    guidelines: str
) -> DebugSuggestion:
    """
    Provide implementation-level debugging suggestions.

    IMPORTANT: This agent provides ONLY implementation-level debugging help.
    It does NOT suggest strategic changes, algorithms, or research ideas.

    Args:
        error_message: Error message from execution
        error_traceback: Full error traceback
        solution_code: The solution code that failed
        problem: Problem description
        requirements: User requirements
        guidelines: Testing contract guidelines

    Returns:
        DebugSuggestion with concise, actionable fix
    """
    llm = get_llm("debugger")

    prompt = f"""Analyze this implementation error and provide a debugging suggestion.

**Problem Context:**
{problem}

**Requirements:**
{requirements}

**Testing Contract (the solution MUST follow this contract):**
{guidelines}

**Solution Code (might be buggy - you need to fix):**
```python
{solution_code}
```

**Error Message:**
{error_message}

**Error Traceback:**
{error_traceback}

---

Analyze the error and provide a SINGLE, CONCISE, ACTIONABLE debugging suggestion (2-4 sentences maximum) that fixes the implementation issue.


---
Some common types of implementation issues to look for:
1. If out of memory errors: it's very likely that the gradient computation graph is too large. Look for operations that incorrectly create large intermediate tensors or retain unnecessary history. 
    - For example: Use torch.autograd.grad() instead of .backward() for cleaner graph management. Properly detach intermediate gradients.
2. If shape mismatches: identify where tensor shapes are incompatible and suggest specific reshaping or transposing operations.
    - For example: Tensor X has shape (10,) but is passed into function expecting (10, 1).
"""

    messages = [
        SystemMessage(content=DEBUGGER_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]
    result = _llm_invoke(llm, DebugSuggestion, messages, "debugger")
    return result


# ============================================================================
# Utility Functions
# ============================================================================
def get_available_gpus() -> list[int]:
    """
    Detect available GPUs using nvidia-smi.

    Returns:
        List of GPU IDs (e.g., [0, 1]) or [] for CPU-only
    """
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            gpu_ids = [int(x.strip()) for x in result.stdout.strip().split("\n") if x.strip()]
            return gpu_ids
    except:
        pass

    # No GPUs detected
    return []


def parse_evaluation_json(stdout: str) -> dict:
    """
    Extract JSON result from evaluate.py stdout.

    The evaluation script prints JSON at the end:
    {"status": "success", "score": 0.00357, "message": "..."}

    Args:
        stdout: Standard output from evaluate.py

    Returns:
        Dictionary with status, score, message

    Raises:
        ValueError: If JSON not found or invalid
    """
    import json
    import re

    # Look for JSON pattern
    json_match = re.search(r'\{"status".*?\}', stdout)
    if not json_match:
        raise ValueError("Could not find JSON output in evaluation stdout")

    result = json.loads(json_match.group())
    return result


# ============================================================================
# Execution Utilities
# ============================================================================

def execute_solution(solution_dir: str, mode: Literal["validate", "train"], parent_id: str = None) -> tuple[bool, str]:
    """
    Execute solution.py with specified mode

    Streams output to train_log.txt in real-time (no terminal output).
    User can monitor progress: tail -f solution_dir/train_log.txt

    Args:
        solution_dir: Path to solution directory
        mode: "validate" (fast test) or "train" (full training)
        parent_id: Parent solution ID (for context)

    Returns:
        (success: bool, log_file_path: str)
    """
    import os
    import sys
    import time
    import threading
    import subprocess
    from constants import TIMEOUT_VALIDATION, TIMEOUT_TRAINING

    log_file = os.path.join(solution_dir, "train_log.txt")

    # Determine timeout based on mode
    timeout = TIMEOUT_VALIDATION if mode == "validate" else TIMEOUT_TRAINING

    try:
        # Set PYTHONUNBUFFERED to force line-buffered output
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'

        # Use Popen for streaming to file
        process = subprocess.Popen(
            [sys.executable, "solution.py", "--mode", mode],
            cwd=solution_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            bufsize=1,  # Line buffered
            env=env
        )

        # Stream output to log file in real-time
        start_time = time.time()
        with open(log_file, 'w') as f:
            while True:
                # Check timeout
                if time.time() - start_time > timeout:
                    process.kill()
                    f.write(f"\n\n!!! TIMEOUT after {timeout} seconds !!!\n")
                    f.flush()
                    return False, log_file

                # Read line from stdout
                line = process.stdout.readline()
                if line:
                    f.write(line)
                    f.flush()  # Ensure immediate write to disk

                # Check if process finished
                if process.poll() is not None:
                    # Read any remaining output
                    remaining = process.stdout.read()
                    if remaining:
                        f.write(remaining)
                        f.flush()
                    break

                # Small sleep if no output to avoid busy waiting
                if not line:
                    time.sleep(0.1)

        returncode = process.returncode
        duration = time.time() - start_time

        # Append return code to log
        with open(log_file, 'a') as f:
            f.write(f"\n\n=== Return code: {returncode} ===\n")

        # Record execution telemetry
        _tel_dir = os.environ.get("SCIML_TELEMETRY_DIR")
        if _tel_dir:
            from telemetry import ExecutionRecord, write_execution_record
            from datetime import datetime as _dt
            write_execution_record(ExecutionRecord(
                timestamp=_dt.utcnow().isoformat(),
                solution_id=os.path.basename(solution_dir),
                iteration=int(os.environ.get("SCIML_TELEMETRY_ITERATION", -1)),
                stage=mode,
                gpu_id=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                duration_seconds=round(duration, 2),
                status="success" if returncode == 0 else "error",
                exit_code=returncode,
            ), _tel_dir)

        if returncode == 0:
            return True, log_file
        else:
            return False, log_file

    except Exception as e:
        # Write error to log file
        with open(log_file, 'w') as f:
            f.write(f"Execution error: {str(e)}\n")
        return False, log_file


def execute_evaluation(solution_dir: str) -> tuple[bool, float, str]:
    """
    Execute evaluate.py and parse results

    Streams output to test_log.txt in real-time (no terminal output).
    User can monitor progress: tail -f solution_dir/test_log.txt

    Args:
        solution_dir: Path to solution directory

    Returns:
        (success: bool, score: float, log_file_path: str)
    """
    import os
    import sys
    import time
    import json
    import subprocess
    from constants import TIMEOUT_EVALUATION

    log_file = os.path.join(solution_dir, "test_log.txt")

    try:
        # Set PYTHONUNBUFFERED to force line-buffered output
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'

        # Use Popen for streaming to file
        process = subprocess.Popen(
            [sys.executable, "evaluate.py"],
            cwd=solution_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            bufsize=1,  # Line buffered
            env=env
        )

        # Stream output to log file in real-time
        start_time = time.time()
        with open(log_file, 'w') as f:
            while True:
                # Check timeout
                if time.time() - start_time > TIMEOUT_EVALUATION:
                    process.kill()
                    f.write(f"\n\n!!! TIMEOUT after {TIMEOUT_EVALUATION} seconds !!!\n")
                    f.flush()
                    return False, float('inf'), log_file

                # Read line from stdout
                line = process.stdout.readline()
                if line:
                    f.write(line)
                    f.flush()  # Ensure immediate write to disk

                # Check if process finished
                if process.poll() is not None:
                    # Read any remaining output
                    remaining = process.stdout.read()
                    if remaining:
                        f.write(remaining)
                        f.flush()
                    break

                # Small sleep if no output to avoid busy waiting
                if not line:
                    time.sleep(0.1)

        returncode = process.returncode
        eval_duration = time.time() - start_time

        # Append return code to log
        with open(log_file, 'a') as f:
            f.write(f"\n\n=== Return code: {returncode} ===\n")

        # Record evaluation telemetry
        _tel_dir = os.environ.get("SCIML_TELEMETRY_DIR")
        if _tel_dir:
            from telemetry import ExecutionRecord, write_execution_record
            from datetime import datetime as _dt
            write_execution_record(ExecutionRecord(
                timestamp=_dt.utcnow().isoformat(),
                solution_id=os.path.basename(solution_dir),
                iteration=int(os.environ.get("SCIML_TELEMETRY_ITERATION", -1)),
                stage="evaluate",
                gpu_id=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                duration_seconds=round(eval_duration, 2),
                status="success" if returncode == 0 else "error",
                exit_code=returncode,
            ), _tel_dir)

        # Read log file to parse JSON score
        with open(log_file, 'r') as f:
            output = f.read()

        # Parse JSON output with robust fallback methods
        # Try primary parsing method (marker-based)
        if "--- FINAL SCALAR METRIC ---" in output:
            try:
                json_str = output.split("--- FINAL SCALAR METRIC ---")[1].strip().split('\n')[0]
                result_json = json.loads(json_str)

                if result_json["status"] == "success":
                    return True, result_json["score"], log_file
                else:
                    # Status is "error" - actual exception occurred
                    return False, float('inf'), log_file
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                print(f"Warning: Failed to parse JSON after marker: {e}")
                # Fall through to backup parsing

        # Backup parsing: Look for JSON anywhere in output using regex
        import re
        json_pattern = r'\{[^}]*"status"[^}]*"score"[^}]*\}'
        matches = re.findall(json_pattern, output, re.DOTALL)
        if matches:
            try:
                # Take the last JSON object found
                result_json = json.loads(matches[-1])
                if result_json.get("status") == "success" and "score" in result_json:
                    return True, result_json["score"], log_file
            except json.JSONDecodeError:
                pass

        # Final fallback: If return code is 0 and no JSON, try to extract score from output
        if returncode == 0:
            # Look for patterns like "Final score: 0.0123" or "Final MSE: 0.0123"
            score_patterns = [
                r'Final score:\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
                r'Final (?:MSE|Loss):\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
                r'Score:\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
            ]
            for pattern in score_patterns:
                match = re.search(pattern, output, re.IGNORECASE)
                if match:
                    score = float(match.group(1))
                    print(f"Warning: Used fallback score extraction: {score}")
                    return True, score, log_file

        # Complete failure - no score found
        print(f"ERROR: Could not extract score from evaluation output")
        print(f"Expected: JSON with '--- FINAL SCALAR METRIC ---' marker")
        print(f"Got: {len(output)} chars, return code: {returncode}")
        print(f"Last 500 chars:\n{output[-500:]}")
        return False, float('inf'), log_file

    except Exception as e:
        # Write error to log file
        with open(log_file, 'w') as f:
            f.write(f"Evaluation error: {str(e)}\n")
        return False, float('inf'), log_file


# ============================================================================
# Proposer and Critic Agents (Ensemble-Guided Batch Mutation)
# ============================================================================

def proposer_agent(
    champion_code: str,
    champion_analysis: str,
    testing_contract: str,
    problem: str,
    requirements: str,
    kb_entry: str,
    ab_reports: str,
    selector_reasoning: str,
    round_num: int,
    conversation_history: list,
    training_set_info: str = "",
    data_analysis_report: str = ""
) -> ProposalOutput:
    """
    Proposer agent that considers both KB and AB context with selector reasoning.

    Combines the conservative (KB-based) and innovative (AB-based) approaches
    into a single agent that leverages both knowledge sources.

    Args:
        champion_code: Parent solution code being mutated (named champion_code for backwards compatibility)
        champion_analysis: Analysis report of parent (named champion_analysis for backwards compatibility)
        testing_contract: Guidelines from testing contract
        problem: Problem description
        requirements: User requirements
        kb_entry: Knowledge base entry (or "No KB entry selected")
        ab_reports: Analysis reports from relatives (parent + siblings + uncles)
        selector_reasoning: Why this solution was selected for mutation
        round_num: Current round number
        conversation_history: Previous messages from debate
        training_set_info: Training dataset information (filename, description, loading instructions)
        data_analysis_report: Optional data analysis report from EDA phase

    Returns:
        ProposalOutput with proposal markdown
    """
    llm = get_llm("proposer")

    # Route to appropriate prompt based on round number
    if round_num < MAX_PROPOSE_CRITIC_ROUNDS - 1:
        # Reasoning mode (rounds 1 to N-2)
        prompt = _proposer_reasoning_prompt(
            champion_code, champion_analysis, testing_contract,
            problem, requirements, kb_entry, ab_reports, selector_reasoning,
            round_num, conversation_history, training_set_info, data_analysis_report
        )
    elif round_num == MAX_PROPOSE_CRITIC_ROUNDS - 1:
        # Synthesis mode (round N-1)
        prompt = _proposer_synthesis_prompt(
            champion_code, champion_analysis, testing_contract,
            problem, requirements, kb_entry, ab_reports, selector_reasoning,
            conversation_history, training_set_info, data_analysis_report
        )
    else:  # round_num == MAX_PROPOSE_CRITIC_ROUNDS
        # Finalization mode (round N)
        prompt = _proposer_finalization_prompt(
            champion_code, champion_analysis, testing_contract,
            problem, requirements, kb_entry, ab_reports, selector_reasoning,
            conversation_history, training_set_info, data_analysis_report
        )

    messages = [
        SystemMessage(content=PROPOSER_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]
    result = _llm_invoke(llm, ProposalOutput, messages, "proposer")
    return result


def _proposer_reasoning_prompt(champion_code, champion_analysis, testing_contract,
                                      problem, requirements, kb_entry, ab_reports,
                                      selector_reasoning, round_num, conversation_history,
                                      training_set_info="", data_analysis_report="") -> str:
    """Generate reasoning prompt for proposer (rounds 1 to N-2)"""

    # Format conversation history if exists
    history_text = ""
    if conversation_history:
        history_text = "\n\n".join([
            f"**{msg.name}**:\n{msg.content}" for msg in conversation_history
        ])
        history_section = f"""
**PREVIOUS DISCUSSION (you should review carefully and reference critically)):**
{history_text}

----

"""
    else:
        history_section = ""

    # Format training data section if exists
    training_section = ""
    if training_set_info:
        training_section = f"""
**TRAINING DATASET:**
{training_set_info}

----

"""

    # Format data analysis section if exists
    data_analysis_section = ""
    if data_analysis_report:
        data_analysis_section = f"""
**DATA ANALYSIS REPORT (insights from training data):**
{data_analysis_report}

----

"""

    return f"""You are a scientific ML research analyst conducting deep theoretical and mathematical exploration.

**SELECTION RATIONALE (why this solution was chosen for mutation):**
{selector_reasoning}

The selector chose this solution based on its potential. Your task is to DEEPLY ANALYZE the problem, not to propose solutions yet.

----

**PROBLEM DESCRIPTION:**
{problem}

**REQUIREMENTS:**
{requirements}

**TESTING CONTRACT (respecting these constraints is critical):**
{testing_contract}

**PARENT SOLUTION CODE:**
```python
{champion_code}
```

**PARENT ANALYSIS:**
{champion_analysis}

----

**KNOWLEDGE BASE CONTEXT (for inspiration and learning):**
{kb_entry}

**ANALYSIS BANK CONTEXT (insights from relatives):**
{ab_reports}

{training_section}{data_analysis_section}----

{history_section}**YOUR TASK - REASONING MODE:**

This is round {round_num} of the reasoning phase. DO NOT propose concrete implementation changes yet. Instead, conduct deep analytical exploration that will help inform a future implementation plan:

1. Start by deeply analyzing the mathematical and theoretical structure of the problem being solved.
- Is this problem governed by differential equations? If so: 
    - What are the equation types, boundary conditions, initial conditions? 
    - What is the domain geometry and dimensionality? Are the geometries complex?
    - What theoretical properties should the solution satisfy? 
    - What numerical challenges exist? (stiffness, stability, convergence, etc.)
    - How is the governing equation enforced? (weak form, strong form, residual minimization, etc.)
    - Is anything singular or pathological about the equations?
- Is this problem learning a mapping from data? If so:
    - What is the data distribution and structure?
    - What are the input/output dimensions and relationships?
    - Is it learning a mapping from function spaces (e.g., operators), or finite-dimensional vectors, or images, or something else?
    - What theoretical properties should the learned mapping satisfy? 
- What alternative mathematical formulations could work?
- Derive or explain key mathematical relationships relevant to this problem
- What does the literature or theory suggest for problems of this type?

**2. Parent Solution Analysis:**
- What is the parent doing?
- What are its strengths, both theoretically and numerically?
- What are its weaknesses or limitations? 
- Think about the following aspects:
    - Model architecture and expressivity
    - Training algorithms and optimization
    - Loss function design and balance
    - Numerical methods and discretization
    - Data sampling and generation
    - Hyperparameter choices 
    (IMPORTANT: the parent's hyperparameters are sometimes too conservative, for example, too few epochs, too small batch sizes, too few data samples, and too small networks. Strongly consider the benefit of using more aggressive hyperparameters) 
- How could the parent be improved, theoretically and numerically?

**3. Knowledge Base and Analysis Bank Insights (if available):**
- What techniques or principles from KB/AB are relevant here?
- How do those techniques work? Are they helpful for THIS specific problem?
- If they are relevant or helpful, you are strongly encouraged to use them. How could they be ADAPTED to this problem? 
    - Example: if KB suggests a DeepONet, then use DeepONet architecture here but adapt input/output dimensions, loss functions, training algorithms, etc. to fit this problem.
    - Don't: if KB suggests a DeepONet, do NOT use FNO. 
- If they are not relevant or helpful, justify and strongly reject them. 

**4. Innovation (you are strongly encouraged to think creatively):**
- Beyond KB/AB, what novel ideas or techniques could be applied here?
- Can you combine multiple ideas in a new way?
- Can you adapt existing methods in a novel way?
- Can you derive new mathematical formulations or algorithms that might work better and are feasible to implement?

**5. Numerical and Computational Considerations:**
- If we use any traditional numerical methods: 
    - Discretization strategies?
    - Stability considerations?
- If we are training a neural network, what architectures are well-suited?
    - Optimization algorithms? Would second-order optimizers or other advanced methods help?
    - Loss functions? Adaptive weight balancing in multi-term losses?
    - Network size and depth? Network architecture innovations? Combining different neural network architectures? 
- Is the training data good enough? 
    - Sampling strategies? Data augmentation? Adaptive sampling?

**PRIORITY HIERARCHY:**
1. Think out loud - show your reasoning process
2. You do NOT need to answer every question above one by one - use them as a guide
3. Strongly consider using insights from the Knowledge Base (KB) and Analysis Base (AB) if relevant
4. If the critic challenged your reasoning, defend or revise your analysis

**OUTPUT FORMAT:**
Generate your analysis in markdown with clear sections. Use mathematical notation where appropriate. This should read like concise research notes, not an implementation plan.

**Remember:** You are having a dialogue with the critic ({get_speaker_name("critic")}). They will challenge your reasoning. Reference their feedback by name if continuing a discussion."""


def _proposer_synthesis_prompt(champion_code, champion_analysis, testing_contract,
                                       problem, requirements, kb_entry, ab_reports,
                                       selector_reasoning, conversation_history,
                                       training_set_info="", data_analysis_report="") -> str:
    """Generate synthesis prompt for round N-1"""

    history_text = "\n\n".join([
        f"**{msg.name}**:\n{msg.content}" for msg in conversation_history
    ])

    training_section = ""
    if training_set_info:
        training_section = f"""
**Training Dataset:**
{training_set_info}
"""

    data_analysis_section = ""
    if data_analysis_report:
        data_analysis_section = f"""
**Data Analysis Report:**
{data_analysis_report}
"""

    return f"""You are transitioning from reasoning to implementation planning. This is the SYNTHESIS round.

**SELECTION RATIONALE:**
{selector_reasoning}

**FULL REASONING DISCUSSION:**
{history_text}

----

**CONTEXT REMINDER:**

**Problem:**
{problem}

**Parent Code:**
```python
{champion_code}
```

**Parent Analysis:**
{champion_analysis}

**Testing Contract:**
{testing_contract}

**Knowledge Base:**
{kb_entry}

**Analysis Bank:**
{ab_reports}
{training_section}{data_analysis_section}
----

**YOUR TASK - SYNTHESIS MODE:**

You've spent the previous rounds conducting deep mathematical and theoretical analysis with the critic ({get_speaker_name("critic")}). Now, synthesize your reasoning into a CONCRETE IMPLEMENTATION PLAN.

**Step 1: Reflect on Reasoning**
- Review the full discussion history above
- What were the key insights from your mathematical analysis?
- What did you and the critic agree on as the most promising approach?
- What theoretical and numerical insights should guide the implementation?

**Step 2: Form Concrete Strategy**
Based on the reasoning, propose a specific implementation approach:
- What concrete changes to make to the parent code?
- What mathematical formulation or algorithm to use?
- What architectural or training modifications?
- How does this connect to the theoretical insights?

**Step 3: Specify Details (START being specific)**
- Provide EXACT hyperparameter values (learning rate, network dims, epochs, etc.)
- Specify exact model architecture components
- Detail data generation or sampling strategies
- Include codes from Knowledge Base if applicable

**CRITICAL - Extract KB Code:**
If the Knowledge Base entry contains relevant code, include the code into a "Reference Code from Knowledge Base" section for the engineer.

**CRITICAL - Avoid Common Pitfalls:**
- Do NOT suggest dropout/regularization unless parent analysis shows clear overfitting
- Do NOT make too many simultaneous changes - keep it focused for observability
- Do NOT use vague language - be precise about what to implement
- If there are multiple possible strategies, choose the BEST one
- If there's no Knowledge Base entry, skip that section and use your own expertise and reasoning

**Priority Hierarchy:**
1. Testing contract requirements (must satisfy)
2. Problem requirements (must satisfy)
3. Knowledge base insights (strongly consider)
4. Selector expectations (strongly consider)
5. You own innovation and theoretical reasoning (strongly consider)
6. Critic suggestions (evaluate critically - you can disagree!)

**OUTPUT FORMAT - Implementation Plan:**

## Summary
(2-3 sentences: What concrete strategy emerged from the reasoning phase?)

## Proposed Changes
(SPECIFIC implementation details - what to change in the code)

## Hyperparameter Recommendations
(**MANDATORY** - List ALL hyperparameters and settings with EXACT values)

## Reference Code from Knowledge Base (if applicable)
(Exact code from KB entry for engineer's reference)

## Implementation Notes
(Practical guidance for the engineer)

**Remember:** You're having a dialogue with {get_speaker_name("critic")}. They will critique this plan. Be ready to defend or refine it."""


def _proposer_finalization_prompt(champion_code, champion_analysis, testing_contract,
                                         problem, requirements, kb_entry, ab_reports,
                                         selector_reasoning, conversation_history,
                                         training_set_info="", data_analysis_report="") -> str:
    """Generate finalization prompt for last round"""

    history_text = "\n\n".join([
        f"**{msg.name}**:\n{msg.content}" for msg in conversation_history
    ])

    training_section = ""
    if training_set_info:
        training_section = f"""
**Training Dataset:**
{training_set_info}
"""

    data_analysis_section = ""
    if data_analysis_report:
        data_analysis_section = f"""
**Data Analysis Report:**
{data_analysis_report}
"""

    return f"""This is the FINAL ROUND. NO FURTHER CRITIQUE will follow. Produce the definitive, implementation-ready proposal.

**SELECTION RATIONALE:**
{selector_reasoning}

**COMPLETE DISCUSSION HISTORY (reasoning → synthesis → plan critique):**
{history_text}

----

**CONTEXT REMINDER:**

**Problem:**
{problem}

**Parent Code:**
```python
{champion_code}
```

**Parent Analysis:**
{champion_analysis}

**Testing Contract:**
{testing_contract}

**Knowledge Base (KB):**
{kb_entry}

**Analysis Bank:**
{ab_reports}
{training_section}{data_analysis_section}
----

**YOUR TASK - FINALIZATION:**

You've completed the full discussion cycle with {get_speaker_name("critic")}:
1. **Reasoning rounds**: Deep mathematical and theoretical analysis
2. **Synthesis round**: Formed concrete implementation plan
3. **Plan critique**: Critic reviewed the implementation plan

Now make your FINAL DECISION. This proposal goes directly to the engineer - there is NO more debate.

**Decision Process:**
1. Review the entire discussion history above
2. Evaluate the critic's feedback on your synthesis plan - were their concerns valid?
3. If multiple strategies were discussed, choose the BEST one
4. Make final refinements based on all feedback
5. Ensure the proposal is complete, specific, and implementation-ready

**If the critic raised concerns in the previous round:**
- Address them if valid, or confidently dismiss if incorrect
- Make your final call - you are empowered to disagree with the critic
- Justify your decision based on requirements and theoretical soundness

**CRITICAL REQUIREMENTS:**
- Include EXACT hyperparameter values (learning rate, network dims, epochs, batch size, etc.)
- Include knowledge base code if applicable (in "Reference Code from Knowledge Base" section)
- Be SPECIFIC and COMPLETE - engineer must implement unambiguously
- Avoid dropout/regularization unless champion shows clear severe overfitting
- Keep changes focused for observability (don't change too many things at once)
- Be CONCISE - prioritize specifics over verbose explanations

**Important implementation notes:**
- This implemenation must work on ONE SHOT and the engineer should not need to ask for clarifications
- The resulted code (which you should NOT write) must run without errors if the engineer follows your plan strictly

**Priority Hierarchy:**
1. Testing contract requirements (must satisfy)
2. Problem requirements (must satisfy)
3. Implementation feasibility and clarity (must be practical)
4. Knowledge base insights (strongly consider)
5. Selector expectations (strongly consider)
6. Critic suggestions (evaluate critically - you can disagree!)

**OUTPUT FORMAT - Implementation Plan:**

## Summary
(2-3 sentences: What concrete strategy emerged from the reasoning phase?)

## Proposed Changes
(SPECIFIC implementation details - what to change in the code)

## Hyperparameter Recommendations
(**MANDATORY** - List ALL hyperparameters and settings with EXACT values)

## Reference Code from Knowledge Base (if applicable)
(Include exact code from KB entry for engineer's reference. MUST cover all necessary components and implementation details.)

## Implementation Notes
(Practical guidance for the engineer)

This is your final word. Make it count."""


def _critic_reasoning_prompt(proposal, champion_code, champion_analysis, problem,
                                    requirements, testing_contract, selector_reasoning,
                                    round_num, conversation_history, data_analysis_report="") -> str:
    """Generate reasoning critique prompt (rounds 1 to N-2)"""

    # Format conversation history
    history_text = ""
    if conversation_history:
        history_text = "\n\n".join([
            f"**{msg.name}**:\n{msg.content}" for msg in conversation_history
        ])
        history_section = f"""
**PREVIOUS DISCUSSION (you should review carefully and reference critically)):**
{history_text}

----

"""
    else:
        history_section = ""

    data_analysis_section = ""
    if data_analysis_report:
        data_analysis_section = f"""
**Data Analysis Report:**
{data_analysis_report}

"""

    return f"""You are a critical evaluator of scientific reasoning and mathematical analysis.

**PROPOSER'S REASONING ANALYSIS:**
{proposal}

----

**SELECTION RATIONALE:**
{selector_reasoning}

{history_section}**CONTEXT:**

**Problem:**
{problem}

**Parent Code:**
```python
{champion_code}
```

**Parent Analysis:**
{champion_analysis}

**Testing Contract:**
{testing_contract}

{data_analysis_section}
----

**YOUR TASK - CRITIQUE REASONING QUALITY:**

This is round {round_num} of the reasoning phase. The proposer ({get_speaker_name("proposer")}) is conducting mathematical and theoretical analysis, NOT proposing implementation yet.

Evaluate the REASONING QUALITY on these dimensions:

**1. Mathematical Rigor:**
- Are the mathematical claims sound?
- Are these claims too general and not specific to the problem (if so, strongly flag it to the proposer)?
- Are these analyses helpful to improving the solution? 

**2. Numerical Understanding:**
- Are these analyses relevant to numerical/computational aspects?
- Are numerical challenges properly identified?
- Are all observations from past numerical behavior considered?
- Are hyperparameter choices analyzed? 
- Will these analyses help improve numerical performance?

**3. KB/AB Integration:**
- Are insights from Knowledge Base and Analysis Bank correctly considered?
- Are there relevant KB/AB insights being missed?
- Are the insights explored deeply for helping with this specific problem?

**4. Data analysis (if data is present):**
- Are insights from data analysis report properly considered?
- Will the reasoning help address data-related challenges, such as noise, distribution, sampling, singularities, etc.?

**5. Innovation**:**
- Is the reasoning innovative and creative?
- Are novel ideas proposed that could lead to breakthroughs? 
- What additional innovation directions could be explored?
- Is it too innovative and is pure intuition and speculation without rigor (if so, strongly flag it to the proposer)?

**You are in dialogue with {get_speaker_name("proposer")}:**
- Reference their reasoning by name when critiquing
- If they addressed your previous concerns, acknowledge it
- If they defended their position, evaluate their defense - they might be right!
- Challenge weak reasoning, and strongly point out obvious mistakes without ambiguity
- Ask probing questions to deepen the analysis (no more than 5)

**CRITICAL:**
1. Do NOT overwhelm with too many possible research directions - focus on the most important ones. 
2. Do NOT be nitpicky - focus on real reasoning quality issues.
3. Do NOT discourage complicated solutions, or express concern regarding model complexity or training cost - we CAN do it and it might work very well! 
4. Do NOT dismiss knowledge base or analysis base insights without strong justification. 
5. If knowledge base solution is considered, it MUST be adapted deeply and specifically to this specific problem - generic references are insufficient.

**OUTPUT FORMAT:**

## Strengths (no more than 5)
What aspects of the reasoning can actually contribute to improving the solution

## Concerns and Flag (no more than 5)
Potential issues or weaknesses (be specific, order them in importance):

## Probing Questions (no more than 5)
Questions to deepen the analysis (help the proposer improve the solution)

## Suggestions (no more than 3)
How to strengthen the reasoning (specific recommendations)

## Potential Plan 
Ideas for concrete implementation directions emerging from the reasoning (very concise, 1-2 sentences)

**Remember:** This is about reasoning quality, NOT implementation feasibility. Save implementation critique for later rounds. Be rigorous but constructive - the goal is to arrive at sound theoretical understanding together."""


def _critic_plan_prompt(proposal, champion_code, champion_analysis, problem,
                               requirements, testing_contract, selector_reasoning,
                               conversation_history, data_analysis_report="") -> str:
    """Generate plan critique prompt (round N-1)"""

    history_text = "\n\n".join([
        f"**{msg.name}**:\n{msg.content}" for msg in conversation_history
    ])

    data_analysis_section = ""
    if data_analysis_report:
        data_analysis_section = f"""
**Data Analysis Report:**
{data_analysis_report}

"""

    return f"""You are a critical evaluator of scientific ML implementation plans.

**IMPLEMENTATION PLAN TO CRITIQUE:**
{proposal}

----

**SELECTION RATIONALE:**
{selector_reasoning}

**FULL DISCUSSION HISTORY:**
{history_text}

----

**CONTEXT:**

**Problem:**
{problem}

**Parent Code:**
```python
{champion_code}
```

**Parent Analysis:**
{champion_analysis}

**Testing Contract:**
{testing_contract}
{data_analysis_section}
----

**YOUR TASK - CRITIQUE IMPLEMENTATION PLAN:**

The proposer ({get_speaker_name("proposer")}) has synthesized the reasoning into a concrete implementation plan. Evaluate it critically:

**1. Alignment with Reasoning:**
- Does the plan follow from the theoretical insights developed in earlier rounds?
- Is it theoretically sound based on the reasoning discussion?
- Are there disconnects between reasoning and implementation?

**2. Specificity:**
- Are EXACT hyperparameter values provided?
- Is the implementation plan unambiguous for the engineer?
- Are there vague or incomplete sections?

**3. KB/AB Integration (if necessary):**
- If KB entry was provided, are key code included for reference?
- Are KB/AB insights properly incorporated?

**4. Performance Potential:**
- Will these changes likely improve the testing metric?
- Is the expected impact realistic?

**5. Requirement Compliance:**
- Does it satisfy testing contract requirements?
- Does it satisfy problem requirements?
- Does it address selector expectations?

**6. Common Pitfalls:**
- Does it suggest premature regularization without overfitting evidence?
- Are network sizes, data sizes, epochs large enough?

**CRITICAL - Dialogue Context:**
- You're having a discussion with {get_speaker_name("proposer")}
- Reference their arguments by name
- If they defended certain choices in the reasoning phase, respect sound justifications
- Challenge weak aspects, acknowledge strong aspects
- They will have one more round to finalize - give constructive feedback

**OUTPUT FORMAT:**

## Strengths (no more than 5)
What the implementation plan does well

## Concerns (no more than 5)
Potential issues or risks (be specific):
- Flag if selector expectations not addressed
- Flag if KB/AB integration is problematic
- Flag if reference codes from KB missing (when KB provided)
- Flag premature regularization without overfitting evidence
- Flag too many simultaneous changes
- Flag vague or incomplete specifications
- Flag requirement violations

## Suggestions (no more than 3)
How to address the concerns (concrete, actionable recommendations)

Be constructive but rigorous. Point out real issues, not nitpicks. The proposer has one more round to finalize."""


def critic_agent(
    proposal: str,
    champion_code: str,
    champion_analysis: str,
    problem: str,
    requirements: str,
    testing_contract: str,
    selector_reasoning: str,
    round_num: int,
    conversation_history: list,
    data_analysis_report: str = ""
) -> CritiqueOutput:
    """
    Critic agent that evaluates proposals considering selector reasoning.

    Args:
        proposal: Current proposal to critique
        champion_code: Champion solution code
        champion_analysis: Champion analysis report
        problem: Problem description
        requirements: User requirements
        testing_contract: Testing guidelines
        selector_reasoning: Why this solution was selected for mutation
        round_num: Current round number
        conversation_history: Previous debate messages
        data_analysis_report: Optional data analysis report from EDA phase

    Returns:
        CritiqueOutput with critique markdown
    """
    llm = get_llm("critic")

    # Route to appropriate prompt based on round number
    if round_num < MAX_PROPOSE_CRITIC_ROUNDS - 1:
        # Reasoning critique (rounds 1 to N-2)
        prompt = _critic_reasoning_prompt(
            proposal, champion_code, champion_analysis, problem,
            requirements, testing_contract, selector_reasoning,
            round_num, conversation_history, data_analysis_report
        )
    else:  # round_num == MAX_PROPOSE_CRITIC_ROUNDS - 1
        # Plan critique (round N-1)
        prompt = _critic_plan_prompt(
            proposal, champion_code, champion_analysis, problem,
            requirements, testing_contract, selector_reasoning,
            conversation_history, data_analysis_report
        )

    messages = [
        SystemMessage(content=CRITIC_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]
    result = _llm_invoke(llm, CritiqueOutput, messages, "critic")
    return result
