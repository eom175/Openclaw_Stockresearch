from pathlib import Path
import sys


def add_project_paths() -> None:
    project_root = Path(__file__).resolve().parents[2]
    src_path = project_root / "src"
    scripts_path = project_root / "scripts"
    for path in (src_path, scripts_path):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
