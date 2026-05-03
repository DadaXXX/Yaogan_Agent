"""Shared utilities for tool modules."""

from pathlib import Path


def safe_path(path: str, base_dir: str) -> Path:
    """Validate that a file path resolves within the allowed base directory.

    Prevents path traversal attacks like ../../etc/passwd.

    Args:
        path: The user-provided file path.
        base_dir: The allowed base directory.

    Returns:
        Resolved Path if safe.

    Raises:
        ValueError: If path escapes the base directory.
    """
    resolved = Path(path).resolve()
    base = Path(base_dir).resolve()

    # Allow paths that are under the base directory
    if resolved.is_relative_to(base):
        return resolved

    # Also allow paths in the current working directory (for input files)
    cwd = Path.cwd().resolve()
    if resolved.is_relative_to(cwd):
        return resolved

    raise ValueError(
        f"路径 '{path}' 不在允许的目录内。"
        f"允许的目录: {base_dir} 或当前工作目录"
    )
