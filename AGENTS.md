# AGENTS.md

## Cursor Cloud specific instructions

### What this repo is
AgenticSciML is a **Python CLI** multi-agent system (no web/UI service). It orchestrates
LLM agents (via LangChain/LangGraph) to generate, critique, and evolve scientific-ML
(PyTorch) solutions. The entry point is `src/main.py`; run everything from inside `src/`.

### Running
```bash
cd src
python3 main.py --help
python3 main.py --mode contract-only   # Phase 1
python3 main.py --mode root-only        # Phase 2
python3 main.py --mode evolve-only      # Phase 3
python3 main.py --mode full             # all phases, non-interactive
```
See `README.md` for the full pipeline description and `src/main_run.sh` for the SLURM example.

### LLM API keys are required to run anything (non-obvious)
Every mode calls an LLM almost immediately, so **no phase runs without provider keys**.
For `full`/`contract-only` a data-analysis phase runs first and is the very first LLM call
(defaults to Gemini), so a missing `GOOGLE_API_KEY` fails before contract creation.
Provide keys either as environment variables or in a `src/.env` file (loaded via
`python-dotenv`):
- `ANTHROPIC_API_KEY` (Claude — tester/engineer/root_engineer)
- `OPENAI_API_KEY` (GPT — critic/debugger + ensemble)
- `GOOGLE_API_KEY` (Gemini — analyst/proposer/retriever/data_analyst + ensemble)
- `XAI_API_KEY` (Grok — ensemble)

The default config (`SCIML_USE_MINI=1`) spreads agents across all four providers, so the
default `full` run needs all four keys. To run with **a single provider's key**, override
the model routing via env vars from `src/constants.py`, e.g. route every agent + the
selection ensemble to OpenAI:
```bash
export SCIML_AGENT_MODEL_TESTER=gpt-5-mini SCIML_AGENT_MODEL_ROOT_ENGINEER=gpt-5-mini \
       SCIML_AGENT_MODEL_ENGINEER=gpt-5-mini SCIML_AGENT_MODEL_ANALYST=gpt-5-mini \
       SCIML_AGENT_MODEL_PROPOSER=gpt-5-mini SCIML_AGENT_MODEL_CRITIC=gpt-5-mini \
       SCIML_AGENT_MODEL_RETRIEVER=gpt-5-mini SCIML_AGENT_MODEL_DEBUGGER=gpt-5-mini \
       SCIML_AGENT_MODEL_DATA_ANALYST=gpt-5-mini SCIML_ENSEMBLE_MODELS=gpt
```
Model IDs / prompts target circa-2025 LLMs; expect prompt/API drift with newer models
(see the README "Note on LLM Versions").

### Tests / lint / build
There is **no automated test suite, no lint config, and no build step** in this repo.
"Verifying" the environment means: imports resolve, `python3 main.py --help` works, and a
mode runs given valid API keys. A ready-to-run function-approximation example ships in
`src/USER_INPUT/` (problem/requirements/evaluation + `train_data.npz`/`val_data.npz`).

### Environment notes
- Dependencies are installed to the user site (`pip install --user`) because
  `python3.12-venv`/`ensurepip` is unavailable on the base image; the update script handles this.
- No GPU is present; GPU auto-detection falls back to CPU (`torch` CPU works fine).
