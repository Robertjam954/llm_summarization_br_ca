# Git Commands Reference

Quick-reference for all common git operations used in this project. On Windows, `git` must be run via Git Bash or by specifying the full path `C:\Program Files\Git\cmd\git.exe` if not in PATH.

---

## Daily Workflow

```powershell
$git = "C:\Program Files\Git\cmd\git.exe"
$repo = "C:\Users\jamesr4\OneDrive - Memorial Sloan Kettering Cancer Center\Documents\GitHub\llm_summarization_br_ca"

# Check status
& $git -C $repo status

# Stage all changes
& $git -C $repo add .

# Stage specific file
& $git -C $repo add notebooks/03_eda_classification_diagnostic_metrics.ipynb

# Commit
& $git -C $repo commit -m "Add element-level diagnostic metrics with bootstrap CIs"

# Push to GitHub
& $git -C $repo push

# Pull latest from GitHub
& $git -C $repo pull
```

---

## Branching

```powershell
# List branches
& $git -C $repo branch -a

# Create and switch to new branch
& $git -C $repo checkout -b feature/prompt-rag-validation

# Switch branch
& $git -C $repo checkout main

# Merge branch into main
& $git -C $repo checkout main
& $git -C $repo merge feature/prompt-rag-validation

# Delete branch after merge
& $git -C $repo branch -d feature/prompt-rag-validation
& $git -C $repo push origin --delete feature/prompt-rag-validation
```

---

## Viewing History

```powershell
# Last 10 commits (one line each)
& $git -C $repo log --oneline -n 10

# Show what changed in last commit
& $git -C $repo show HEAD

# Show diff of unstaged changes
& $git -C $repo diff

# Show diff of staged changes
& $git -C $repo diff --cached

# Who changed what in a file
& $git -C $repo log --follow -p notebooks/03_eda_classification_diagnostic_metrics.ipynb
```

---

## Undoing Changes

```powershell
# Discard unstaged changes to a file (IRREVERSIBLE)
& $git -C $repo restore notebooks/01_deidentification.ipynb

# Unstage a file (keep changes)
& $git -C $repo restore --staged notebooks/01_deidentification.ipynb

# Undo last commit but keep changes staged
& $git -C $repo reset --soft HEAD~1

# Undo last commit and unstage (keep changes in working dir)
& $git -C $repo reset HEAD~1

# Revert a pushed commit (creates a new "undo" commit — safe for shared branches)
& $git -C $repo revert <commit-hash>
```

---

## Managing Tracked Files

```powershell
# Stop tracking a file/folder (keep local copy)
& $git -C $repo rm --cached references/some_file.pdf
& $git -C $repo rm --cached -r references/

# Remove file from repo AND local disk
& $git -C $repo rm references/some_file.pdf

# Check what is currently tracked
& $git -C $repo ls-files
```

---

## Remote Operations

```powershell
# List remotes
& $git -C $repo remote -v

# Fetch without merging
& $git -C $repo fetch origin

# Pull with rebase (cleaner history)
& $git -C $repo pull --rebase origin main

# Force push (use with caution — rewrites remote history)
& $git -C $repo push --force-with-lease
```

---

## Stashing

```powershell
# Save work in progress without committing
& $git -C $repo stash push -m "wip: NB07 bert batch size tweak"

# List stashes
& $git -C $repo stash list

# Apply most recent stash (keep stash)
& $git -C $repo stash apply

# Apply and drop stash
& $git -C $repo stash pop

# Drop a specific stash
& $git -C $repo stash drop stash@{0}
```

---

## Tagging Releases

```powershell
# Create annotated tag
& $git -C $repo tag -a v1.0 -m "Initial analysis complete — ACS abstract submission"

# Push tags
& $git -C $repo push origin --tags

# List tags
& $git -C $repo tag
```

---

## .gitignore Quick Reference

```gitignore
# Ignore entire folder
references/

# Ignore file type
*.pdf
*.xlsx

# Ignore but NOT a specific file
!.env.example
*.env

# Ignore nested folder anywhere
**/__pycache__/
**/.ipynb_checkpoints/
```

After updating `.gitignore`, untrack already-committed files:
```powershell
& $git -C $repo rm --cached -r <folder-or-file>
& $git -C $repo add .gitignore
& $git -C $repo commit -m "Remove <folder> from tracking, update .gitignore"
& $git -C $repo push
```

---

## Git Config (One-Time Setup)

```powershell
& "C:\Program Files\Git\cmd\git.exe" config --global user.name "Robertjam954"
& "C:\Program Files\Git\cmd\git.exe" config --global user.email "your-msk-email@mskcc.org"
& "C:\Program Files\Git\cmd\git.exe" config --global core.autocrlf true
& "C:\Program Files\Git\cmd\git.exe" config --global init.defaultBranch main

# Shorten git to an alias in PowerShell profile
# Add to $PROFILE:
Set-Alias git "C:\Program Files\Git\cmd\git.exe"
```

Add to PowerShell profile (`notepad $PROFILE`):
```powershell
Set-Alias git "C:\Program Files\Git\cmd\git.exe"
```

Then `git` will work without the full path in any future PowerShell session.
