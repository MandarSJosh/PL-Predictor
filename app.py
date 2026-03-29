"""
Streamlit entrypoint at repo root (Streamlit Community Cloud default: streamlit run app.py).
Delegates to the main dashboard module.
"""
from pathlib import Path
import runpy

_ROOT = Path(__file__).resolve().parent
_DASH = _ROOT / "src" / "dashboard" / "app.py"
runpy.run_path(str(_DASH), run_name="__main__")
