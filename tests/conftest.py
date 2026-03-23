from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DECT_INIT = PROJECT_ROOT / "dect" / "__init__.py"

# Ensure pytest resolves imports against this repository first.
# The editable install .pth file adds PROJECT_ROOT to sys.path but after
# site-packages, so the pip-installed 'dect' would shadow the local one.
# Always move it to position 0.
project_root_str = str(PROJECT_ROOT)
if project_root_str in sys.path:
    sys.path.remove(project_root_str)
sys.path.insert(0, project_root_str)

loaded = sys.modules.get("dect")
if loaded is not None:
    loaded_file = getattr(loaded, "__file__", "")
    if loaded_file:
        try:
            loaded_path = Path(loaded_file).resolve()
            if loaded_path != LOCAL_DECT_INIT.resolve():
                del sys.modules["dect"]
        except OSError:
            del sys.modules["dect"]
