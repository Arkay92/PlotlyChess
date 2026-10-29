"""Plotly Cloud persistent filesystem discovery and database path resolution.

Plotly Cloud allows attaching persistent network filesystems (such as 'Games')
to Dash applications. Storage in this directory survives container restarts,
worker cycling, and application redeployments.

Reference: https://plotly.com/blog/plotly-cloud-persistent-filesystems/
"""
import os
from pathlib import Path


def _is_usable_directory(candidate: Path, create_if_parent_exists: bool = False) -> bool:
    """Check if a path is (or can be made into) a writable directory."""
    try:
        if candidate.is_dir():
            test_file = candidate / ".plotly_storage_write_test"
            test_file.touch()
            test_file.unlink()
            return True
        if create_if_parent_exists and candidate.parent.is_dir():
            candidate.mkdir(parents=True, exist_ok=True)
            test_file = candidate / ".plotly_storage_write_test"
            test_file.touch()
            test_file.unlink()
            return True
    except (OSError, PermissionError):
        return False
    return False


def get_cloud_storage_dir(name: str = "Games") -> Path | None:
    """Locate the Plotly Cloud persistent filesystem directory.

    Checks in order:
    1. Explicit environment variables:
       - GAMES_DIR (or <NAME>_DIR)
       - DASH_DATA_DIR (standard Dash Enterprise / Plotly variable)
       - PERSISTENT_DIR, DATA_DIR, PLOTLY_STORAGE_DIR, STORAGE_DIR
    2. Cloud container mount points:
       - /<name> (e.g. /Games)
       - /<name_lower> (e.g. /games)
       - /data/<name>, /data/<name_lower>, /data
       - /mnt/<name>, /mnt/<name_lower>, /mnt/data
       - ../mount/<name>, ../mount (standard Dash Enterprise persistent mount)
       - ./<name>, ./<name_lower> (relative mount in workspace)

    Returns:
        Path to the verified writable persistent storage directory,
        or None if no cloud persistent filesystem is detected (e.g. local dev, unit tests).
    """
    clean_name = (name or "Games").strip()
    name_upper = clean_name.upper()
    name_lower = clean_name.lower()

    # 1. Check environment variables
    env_vars = (
        f"{name_upper}_DIR",
        "DASH_DATA_DIR",
        "PERSISTENT_DIR",
        "DATA_DIR",
        "PLOTLY_STORAGE_DIR",
        "STORAGE_DIR",
    )
    for env_key in env_vars:
        val = os.environ.get(env_key)
        if val and val.strip():
            candidate = Path(val.strip()).resolve()
            if _is_usable_directory(candidate, create_if_parent_exists=True):
                return candidate
            sub_candidate = candidate / clean_name
            if _is_usable_directory(sub_candidate, create_if_parent_exists=True):
                return sub_candidate

    # 2. Check standard Linux/Cloud mount paths (only if directory already exists)
    app_root = Path(__file__).resolve().parent.parent
    cwd = Path.cwd()

    mount_candidates = [
        Path(f"/{clean_name}"),
        Path(f"/{name_lower}"),
        Path(f"/data/{clean_name}"),
        Path(f"/data/{name_lower}"),
        Path("/data"),
        Path(f"/mnt/{clean_name}"),
        Path(f"/mnt/{name_lower}"),
        Path("/mnt/data"),
        cwd.parent / "mount" / clean_name,
        cwd.parent / "mount",
        app_root / clean_name,
        app_root / name_lower,
        cwd / clean_name,
        cwd / name_lower,
    ]

    for candidate in mount_candidates:
        try:
            # We only adopt pre-existing directories from system mount candidates
            if candidate.is_dir() and _is_usable_directory(candidate, create_if_parent_exists=False):
                return candidate.resolve()
        except (OSError, PermissionError):
            continue

    return None


def get_cloud_db_paths(name: str = "Games") -> tuple[Path | None, Path | None]:
    """Return (rooms_db_path, users_db_path) if persistent cloud storage is detected."""
    storage_dir = get_cloud_storage_dir(name=name)
    if storage_dir is None:
        return None, None
    return (
        storage_dir / "plotly_chess_rooms.sqlite3",
        storage_dir / "plotly_chess_users.sqlite3",
    )
