# Jupyter Notebook Setup in VSCode

---

## Quick Start: Open or Create a Notebook

**Open existing notebook:**
File → Open File → select any `.ipynb` in `notebooks/`

**Create new notebook:**
`Ctrl+Shift+P` → *Jupyter: Create New Jupyter Notebook* → save to `notebooks/`

**Select kernel:**
Click the kernel picker (top right of notebook) → select `.venv (Python 3.x)` or your conda env.

---

## Kernel Registration (One-Time Per Environment)

If your project env doesn't appear in the kernel list:

```powershell
# Activate env first
.venv\Scripts\Activate.ps1

# Install and register kernel
pip install ipykernel
python -m ipykernel install --user --name=llm_br_ca --display-name "Python (llm_br_ca)"

# Verify it appears
jupyter kernelspec list
```

**Conda env variant:**
```powershell
conda activate llm_br_ca
conda install ipykernel
python -m ipykernel install --user --name=llm_br_ca --display-name "Python (llm_br_ca)"
```

---

## VSCode Settings for Notebooks

Paste into `.vscode/settings.json` (workspace settings, not committed to git):

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/Scripts/python.exe",
  "jupyter.notebookFileRoot": "${workspaceFolder}",
  "jupyter.interactiveWindow.textEditor.executeSelection": true,
  "notebook.formatOnSave.enabled": true,
  "notebook.codeActionsOnSave": {
    "notebook.source.fixAll": "explicit",
    "notebook.source.organizeImports": "explicit"
  },
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true
  }
}
```

> Source: `vscode_config/jupyter templates.txt`

---

## nbautoexport — Auto-Export Notebooks to `.py` on Save

Automatically generates a `.py` script alongside each `.ipynb` on every save — useful for code review and git diffs.

**Install (once per machine/env):**
```bash
pip install nbautoexport
nbautoexport install
```

**Configure for this project's notebooks folder:**
```bash
cd notebooks/
nbautoexport configure
```
Creates a `.nbautoexport` config file in `notebooks/`. After this, every `Ctrl+S` in a notebook writes a matching `.py` to `notebooks/script/`.

**Manual export:**
```bash
nbautoexport export notebooks/03_eda_classification_diagnostic_metrics.ipynb
# or export all at once:
nbautoexport export notebooks/
```

> Source: `vscode_config/nbautoexport_export jupyter notebook to script.rtf`

---

## Workflow: Anaconda Environment with Jupyter (Original Notes)

From `vscode_config/vs code and jupyter notebook config on open.rtf`:

1. Open VSCode → **File → Open Folder** (project root)
2. `Ctrl+Shift+P` → **Python: Create Virtual Environment** → select **Anaconda**
3. `Ctrl+Shift+P` → **Python: Activate Environment**
4. `Ctrl+Shift+P` → **Jupyter: Create New Jupyter Notebook**
5. Select kernel → **Python 3** (from the activated conda env)

---

## Conda Environment: Local (Prefix) Setup

From `vscode_config/conda env local creation and activation.txt`:

```powershell
# Create env in project subdirectory
conda create --prefix ./envs python=3.11 jupyterlab pandas numpy

# Activate by prefix
conda activate ./envs

# Shorten prompt (edit .condarc)
conda config --set env_prompt '({name})'
```

Benefits of prefix envs:
- All project dependencies live inside the project folder
- No name collisions across projects
- Portable — move the folder and the env moves with it

---

## Jupyter Lab (Standalone) Kernel Setup

If running JupyterLab outside VSCode:

```powershell
# Install JupyterLab into env
pip install jupyterlab

# Register kernel (same as VSCode)
python -m ipykernel install --user --name=llm_br_ca --display-name "Python (llm_br_ca)"

# Launch
jupyter lab
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Kernel doesn't appear in picker | Re-run `python -m ipykernel install --user ...`, restart VSCode |
| `import` fails but package is installed | Wrong kernel selected — check that interpreter path matches the env |
| Notebook output not clearing on commit | Add `*.ipynb` output stripping: `nbstripout --install` |
| `load_dotenv()` doesn't find `.env` | Ensure `jupyter.notebookFileRoot` is `${workspaceFolder}` in settings |
| Ruff formatter not applying | Install Ruff extension; confirm `"editor.defaultFormatter": "charliermarsh.ruff"` |
