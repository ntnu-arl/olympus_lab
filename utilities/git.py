import os
import subprocess
from pathlib import Path


import os
import subprocess
from pathlib import Path
from datetime import datetime


def track_experiment(tag):
    # 1. Find the Git root
    repo_root = _find_git_root()

    if _branch_exists(tag, cwd=repo_root):
        raise RuntimeError(f"Branch '{tag}' already exists. Please choose a different tag.")

    is_clean = _is_working_tree_clean(cwd=repo_root)

    if not is_clean:
        # 2. Save list of staged files
        staged_files = _run_git("git diff --name-only --cached", cwd=repo_root).splitlines()

        # 3. Stage everything and commit
        _run_git("git add .", cwd=repo_root)
        _run_git(f'git commit -m "WIP: RL experiment {tag}"', cwd=repo_root)

    # 4. Create the branch (without switching)
    _run_git(f"git branch {tag}", cwd=repo_root)

    if not is_clean:

        # 5. Reset the commit (keep all changes staged)
        _run_git("git reset --soft HEAD~1", cwd=repo_root)

        # 6. Unstage everything
        _run_git("git reset", cwd=repo_root)

        # 7. Re-stage only originally staged files
        for file in staged_files:
            if file:  # ignore empty strings
                _run_git(f"git add {file}", cwd=repo_root)


# Example usage


def _is_working_tree_clean(cwd):
    """Return True if there are no staged or unstaged changes."""
    unstaged = _run_git("git diff --name-only", cwd=cwd)
    staged = _run_git("git diff --name-only --cached", cwd=cwd)
    return not unstaged.strip() and not staged.strip()


def _branch_exists(branch_name, cwd):
    return bool(_run_git(f"git branch --list {branch_name}", cwd=cwd).strip())


def _find_git_root(start_path="."):
    path = Path(start_path).resolve()
    for parent in [path] + list(path.parents):
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Not inside a Git repository.")


def _run_git(cmd, cwd=None):
    """Run a git command and return its output (if any)."""
    result = subprocess.run(cmd, shell=True, check=True, cwd=cwd, stdout=subprocess.PIPE)
    return result.stdout.decode().strip()


if __name__ == "__main__":
    experiment_tag = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    track_experiment(experiment_tag)
