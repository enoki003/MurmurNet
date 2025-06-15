import os
from pathlib import Path
from typing import Optional

def get_project_root() -> Path:
    # Assuming path_utils.py is in murmurnet/modules/
    # Adjust if the file location is different.
    return Path(__file__).resolve().parent.parent.parent

def resolve_path(path_str: str, project_root_override: Optional[Path] = None) -> str:
    if not path_str:
        return ""

    path = Path(path_str)

    # Expand user home directory
    if str(path).startswith('~'):
        return str(path.expanduser())

    # Check if path is absolute
    if path.is_absolute():
        return str(path)

    # Resolve relative to project root
    root = project_root_override if project_root_override else get_project_root()
    resolved = root / path
    return str(resolved.resolve())
