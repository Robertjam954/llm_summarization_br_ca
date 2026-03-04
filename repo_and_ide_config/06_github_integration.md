# GitHub Integration

Reference for authenticating to GitHub, working with the remote repo, and using GitHub features from VSCode and PowerShell.

---

## Authentication

### Personal Access Token (PAT) — Recommended for Windows

1. Go to [github.com/settings/tokens](https://github.com/settings/tokens) → **Tokens (classic)** → **Generate new token**
2. Set scopes: `repo` (full), `workflow`, `read:org`
3. Copy the token (shown only once)
4. Store it in `.env` as `GITHUB_PAT=ghp_...` (never commit)

**Use PAT for HTTPS push from PowerShell:**
```powershell
$git = "C:\Program Files\Git\cmd\git.exe"
$repo = "C:\Users\jamesr4\OneDrive - ...\llm_summarization_br_ca"
$pat  = (Get-Content .env | Select-String "GITHUB_PAT").ToString().Split("=")[1].Trim()

& $git -C $repo remote set-url origin "https://$pat@github.com/Robertjam954/llm_summarization_br_ca.git"
& $git -C $repo push
```

### Windows Credential Manager (Simpler)
On first `git push`, Windows will prompt for GitHub login. Use your GitHub username + PAT as the password. The credential is cached in Windows Credential Manager.

To update a stored credential:
**Control Panel → Credential Manager → Windows Credentials → github.com → Edit**

### SSH Key (Alternative)
```powershell
# Generate key
ssh-keygen -t ed25519 -C "your-email@mskcc.org"

# Copy public key
Get-Content ~/.ssh/id_ed25519.pub | clip

# Add to GitHub: github.com/settings/keys → New SSH Key → paste
# Change remote to SSH
& $git -C $repo remote set-url origin git@github.com:Robertjam954/llm_summarization_br_ca.git
```

---

## Repository Info

| Property | Value |
|---|---|
| Remote URL (HTTPS) | `https://github.com/Robertjam954/llm_summarization_br_ca.git` |
| Remote URL (SSH) | `git@github.com:Robertjam954/llm_summarization_br_ca.git` |
| Default branch | `main` |
| Visibility | Private |

---

## VSCode GitHub Integration

### GitLens Extension
- **Source Control panel** (`Ctrl+Shift+G`): stage, commit, push, pull
- **GitLens Launchpad** (`Ctrl+Shift+P` → *GitLens: Open Launchpad*): shows open PRs and their review status
- **Commit Composer** (`Ctrl+Shift+P` → *GitLens: Open Commit Graph*): visual commit history
- **Inline blame**: shows last commit per line in the editor

### GitHub Pull Requests Extension
- View and review PRs without leaving VSCode
- Install: `GitHub Pull Requests` by GitHub
- `Ctrl+Shift+P` → *GitHub Pull Requests: Create Pull Request*

---

## Creating a Pull Request via PowerShell

```powershell
# Using GitHub CLI (gh)
winget install GitHub.cli   # one-time install
gh auth login               # follow prompts

$repo_dir = "C:\Users\jamesr4\OneDrive - ...\llm_summarization_br_ca"
Set-Location $repo_dir

gh pr create `
  --title "Add NB07 validation methods comparison results" `
  --body "Adds XGBoost + BERT validation with SHAP feature ranking. Closes #12." `
  --base main `
  --head feature/nb07-validation
```

---

## GitHub Actions (CI)

The repo does not currently have CI configured. To add basic Python linting on push:

Create `.github/workflows/lint.yml`:
```yaml
name: Lint

on: [push, pull_request]

jobs:
  ruff:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install ruff
      - run: ruff check notebooks/ src/
```

---

## Branch Protection (Recommended)

On GitHub: **Settings → Branches → Add rule** for `main`:
- ✅ Require pull request reviews before merging
- ✅ Require status checks to pass
- ✅ Restrict who can push to matching branches

---

## Useful GitHub CLI Commands

```powershell
# View repo status
gh repo view

# List open issues
gh issue list

# Create issue
gh issue create --title "Fix NB03 bootstrap CI calculation" --body "..."

# List PRs
gh pr list

# Check out a PR locally
gh pr checkout 5

# View CI run status
gh run list
gh run view <run-id>

# Clone repo (alternative to git clone)
gh repo clone Robertjam954/llm_summarization_br_ca
```

---

## Colab → GitHub Push

When pushing from Google Colab, use a PAT stored in Colab Secrets:

```python
from google.colab import userdata
import subprocess, os

pat = userdata.get("GITHUB_PAT")
repo_path = "/content/llm_summarization_br_ca"

subprocess.run(["git", "config", "user.email", "your-email@mskcc.org"], cwd=repo_path)
subprocess.run(["git", "config", "user.name", "Robertjam954"], cwd=repo_path)
subprocess.run(["git", "remote", "set-url", "origin",
    f"https://{pat}@github.com/Robertjam954/llm_summarization_br_ca.git"], cwd=repo_path)
subprocess.run(["git", "add", "reports/", "data/features/"], cwd=repo_path)
subprocess.run(["git", "commit", "-m", "Add Colab analysis outputs"], cwd=repo_path)
subprocess.run(["git", "push"], cwd=repo_path)
```

> See `docs/colab_pipeline_guide.md` → Part 5 for the full version.

---

## Fix: `git` Not in PowerShell PATH

Add to your PowerShell profile (`notepad $PROFILE`):
```powershell
Set-Alias git "C:\Program Files\Git\cmd\git.exe"
```

Or set via System Environment Variables:
**Start → Edit the system environment variables → Environment Variables → Path → New** → add `C:\Program Files\Git\cmd`

Then restart VSCode/PowerShell.
