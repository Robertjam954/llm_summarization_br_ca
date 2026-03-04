# VSCode Startup Guide — Every Session Checklist

Quick-start reference for opening this project in VSCode and getting Python/Jupyter ready.

---

## Every Session: 5-Step Startup

### 1. Open the Workspace Folder
File → Open Folder → select:
```
C:\Users\jamesr4\OneDrive - Memorial Sloan Kettering Cancer Center\Documents\GitHub\llm_summarization_br_ca
```
Or use the `.code-workspace` file if present (double-click opens the full workspace).

### 2. Activate the Project Environment

**Option A — venv (recommended for this project):**
```powershell
# In VSCode integrated terminal (Ctrl + `)
.venv\Scripts\Activate.ps1
```

**Option B — conda/anaconda:**
```powershell
conda activate ./envs
# or by name:
conda activate llm_br_ca
```

Confirm activation — your terminal prompt should show `(.venv)` or `(llm_br_ca)`.

### 3. Select Python Interpreter
`Ctrl+Shift+P` → **Python: Select Interpreter** → pick the `.venv` or `envs` entry that shows the project path.

If the env doesn't appear, paste the full path manually:
```
C:\Users\jamesr4\OneDrive - ...\llm_summarization_br_ca\.venv\Scripts\python.exe
```

### 4. Load Environment Variables
The project uses `.env` for all path and API key config. Confirm `.env` exists:
```powershell
Test-Path .env   # should return True
```
If missing: `Copy-Item .env.example .env` then edit values.

### 5. Verify Setup
```powershell
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.getenv('PROJECT_ROOT'))"
```
Should print: `C:\Users\jamesr4\OneDrive - ...\llm_summarization_br_ca`

---

## First-Time Project Setup

```powershell
# Clone
git clone https://github.com/Robertjam954/llm_summarization_br_ca.git
cd llm_summarization_br_ca

# Create venv
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Configure paths
Copy-Item .env.example .env
# Edit .env: set PROJECT_ROOT and DATA_PRIVATE_DIR
```

---

## Useful VSCode Keyboard Shortcuts

| Action | Shortcut |
|---|---|
| Command Palette | `Ctrl+Shift+P` |
| Toggle terminal | `` Ctrl+` `` |
| Open settings (JSON) | `Ctrl+Shift+P` → *Open User Settings (JSON)* |
| Select interpreter | `Ctrl+Shift+P` → *Python: Select Interpreter* |
| New Jupyter notebook | `Ctrl+Shift+P` → *Jupyter: Create New Jupyter Notebook* |
| Run all notebook cells | `Ctrl+Shift+P` → *Jupyter: Run All Cells* |
| Format on save | Auto via Ruff (see `vscode_config/jupyter templates.txt`) |

---

## Recommended Extensions for This Project

| Extension | Purpose |
|---|---|
| Python (Microsoft) | Interpreter, IntelliSense, linting |
| Jupyter (Microsoft) | Notebook editing inside VSCode |
| Ruff | Fast Python linter + formatter |
| GitLens | Enhanced git blame, history, Launchpad |
| GitHub Copilot | AI code completion |
| Windsurf / Cascade | Agentic AI coding assistant |
| Data Wrangler | DataFrame preview inside VSCode |
| Great Icons | File icon theme |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Terminal doesn't activate `.venv` | Run: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Wrong Python version selected | `Ctrl+Shift+P` → *Python: Clear Cache and Reload Window* |
| `dotenv` not found | `pip install python-dotenv` |
| Jupyter kernel missing | See `02_jupyter_notebook_vscode.md` |
| Git not recognized in terminal | Add `C:\Program Files\Git\cmd` to PATH or use Git Bash terminal |
