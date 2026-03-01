# Environment Setup Guide

This document walks through the full workflow for getting the `llm_summarization` repository ready for development in VS Code on macOS, from opening the project to running scripts with the local Python interpreter.

## 1. Prerequisites

- macOS with Command Line Tools installed (Xcode Command Line Tools or full Xcode)
- Homebrew (optional but helpful for installing system packages)
- **Python 3.12.x recommended.** The default macOS `python3` (currently 3.14) works for most libraries, but packages such as `arize-phoenix-otel` require `<3.14`. Install Python 3.12 via Homebrew (`brew install python@3.12`) or `pyenv` and ensure it is on your PATH.
- Visual Studio Code with the official Python extension enabled

## 2. Open the project folder in VS Code

1. Launch VS Code.
2. Use **File → Open Folder…** and select `/Users/robertjames/Documents/llm_summarization`.
3. VS Code will reload with the project Explorer showing folders like `scripts_notebooks/`, `data/`, etc.

## 3. Save (or update) the workspace file

If you want a reusable VS Code workspace file:

1. Go to **File → Save Workspace As…**.
2. Save it as `llm_summarization.code-workspace` inside the project root (this already exists—overwrite it if you have new folders/settings to capture).
3. Next time you can open the workspace by double-clicking the `.code-workspace` file or via **File → Open Workspace from File…**.

## 4. Create a dedicated virtual environment

1. Open a terminal (integrated VS Code terminal or macOS Terminal).
2. Make sure you are at the project root:
   ```bash
   cd /Users/robertjames/Documents/llm_summarization
   ```
3. Create the environment (only needed once). Use the Python 3.12 interpreter for maximum package compatibility:
   ```bash
   python3.12 -m venv .venv
   ```
   This produces the folder `.venv/` with an isolated interpreter.

## 5. Activate the virtual environment (macOS zsh)

Every new terminal session needs activation before running Python commands:
```bash
source .venv/bin/activate
```
You should see `(.venv)` prefixed in the shell prompt. To exit later, run `deactivate`.

## 6. Install VS Code Python interpreter binding

1. With the env activated, open the VS Code Command Palette (`⇧⌘P`).
2. Run **Python: Select Interpreter**.
3. Choose the interpreter pointing to `.venv/bin/python`. If it’s missing, pick **Enter interpreter path…** and browse to `/Users/robertjames/Documents/llm_summarization/.venv/bin/python`.
4. VS Code now uses this interpreter for terminals, notebooks, and linting.

## 7. Manage dependencies with `requirements.txt`

The repo now includes `requirements.txt` with the core libraries:

```
pandas>=2.1
numpy>=1.26
statsmodels>=0.14
plotly>=5.18
matplotlib>=3.8
deepeval>=0.20
pydantic>=2.7
```

To install (or reinstall) everything inside the active virtual environment:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### When you need `arize-phoenix-otel` (or any package requiring <3.14)

1. Remove or rename the existing `.venv` if it was built with Python 3.14:
   ```bash
   rm -rf .venv
   ```
2. Recreate the virtual environment using Python 3.12 explicitly (adjust the path if you installed via `pyenv`):
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   ```
3. Reinstall baseline dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. Install the extra package:
   ```bash
   pip install arize-phoenix-otel pip install arize-phoenix-otel
   ```
5. (Optional) Append `arize-phoenix-otel` to `requirements.txt` if it should be part of the default stack.

### Adding new packages
- Append the package (and an optional version specifier) to `requirements.txt`.
- Re-run `pip install -r requirements.txt` to sync the environment.
- Commit the updated file so collaborators can reproduce the same setup.

## 8. Configure environment variables (if needed)

Models or judge APIs often require API keys (OpenAI, Anthropic, Phoenix, etc.). Add them to your shell session before running scripts:
```bash
export OPENAI_API_KEY="sk-proj-YOUR_KEY_HERE"
export ANTHROPIC_API_KEY="sk-ant-YOUR_KEY_HERE"
export PHOENIX_API_KEY="YOUR_PHOENIX_API_KEY_HERE"
export PHOENIX_COLLECTOR_ENDPOINT="https://app.phoenix.arize.com/s/YOUR_ENDPOINT"
```

If you prefer to keep secrets out of your shell history, create a `.env` file inside the project root (remember to add `.env` to `.gitignore`) and store them there:
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=...
PHOENIX_API_KEY=<your-api-key>
PHOENIX_COLLECTOR_ENDPOINT=https://app.phoenix.arize.com/s/robertjam954
```
Then load them automatically by either:

- Installing/using `python-dotenv` to read `.env` at runtime, or
- Enabling VS Code’s **Python › Terminal: Activate Env File** setting and pointing it to `${workspaceFolder}/.env`, or
- Sourcing the file manually (`set -a; source .env; set +a`).

However you set them, confirm they’re available before running Phoenix instrumentation:
```bash
echo $PHOENIX_API_KEY
echo $PHOENIX_COLLECTOR_ENDPOINT
```

## 9. Verify the installation

Run a quick import smoke test to confirm that the interpreter can load the dependencies:
```bash
python - <<'PY'
import pandas, numpy, statsmodels.api as sm
import plotly, matplotlib, deepeval, pydantic
print("imports-ok")
PY
```
You should see `imports-ok` with no stack trace.

## 10. Run project scripts or notebooks

### Python scripts
With the environment activated:
```bash
python scripts_notebooks/llm_analysis_primary_metrics.py
```
Replace the filename with any other script you want to execute.

### Jupyter/VS Code notebooks
1. Open a `.ipynb` file (e.g., `scripts_notebooks/needle_haystack_context length llm accuracy visualization.ipynb`).
2. When prompted, select the `.venv` interpreter/kernel.
3. Run cells normally; all installed packages are available.

## 11. Keeping tooling consistent

- When switching machines, repeat steps 4–10.
- After pulling new code, re-run `pip install -r requirements.txt` to pick up dependency changes.
- If VS Code forgets the interpreter, repeat step 6.

Following the steps above ensures the local environment stays reproducible and ready for running any scripts, notebooks, or experiments in this repository.
