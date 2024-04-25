import os
from git import Repo, InvalidGitRepositoryError, GitError
from pathlib import Path
from typing import Dict
import logging

IGNORE_REPO_FILENAME = "DONT_TRACK"

logger = logging.getLogger(__name__)


class UncommittedChangesError(GitError):
    pass


def track_git_repos(working_path: Path | None = None) -> Dict[str, str]:
    """
    Tracks Git repositories within a working directory and logs their latest commit hashes.

    Raises `UncommittedChangesError` if any repositories have uncommitted changes.

    Repositories can be excluded from version checking by creating an empty file named
    "DONT_TRACK" in their subfolders.

    Args:
        working_path (pathlib.Path, optional): The starting path. If None, the
            current directory is assumed.

    Returns:
        A dictionary mapping repository folder names (str) to their latest commit hashes (str).
    """

    git_repos_commits = {}
    if working_path is None:
        working_path = _get_current_working_dir()
    for p in working_path.rglob("*.git"):
        p = p.parent
        if not p.is_dir():
            continue

        try:
            repo = Repo(p)
        except InvalidGitRepositoryError:
            logger.warning(f"{p} is not a git repository, but contains a .git subfolder. Skipping.")
            continue

        if (p / IGNORE_REPO_FILENAME).is_file():
            logger.warning(f"{p} is a git repository, but contains the {IGNORE_REPO_FILENAME} file. Skipping.")
            continue

        if repo.is_dirty():
            raise UncommittedChangesError(f"The repository {p} have uncomitted changes.")
        git_repos_commits[p.name] = repo.head.commit.hexsha

    return git_repos_commits


def _get_current_working_dir() -> Path:
    return Path(os.getcwd())
