from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DECT_INIT = PROJECT_ROOT / "dect" / "__init__.py"

# Ensure pytest resolves imports against this repository first.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
