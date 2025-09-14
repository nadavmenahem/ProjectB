from pathlib import Path

def find_repo_root(start: Path) -> Path:
    """Walk upwards until we hit a folder that contains .git or pyproject.toml."""
    for parent in [start, *start.parents]:
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not find repository root.")