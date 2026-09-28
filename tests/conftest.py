import sys
from pathlib import Path

# CLI scripts (install_mobile_bundle.py, ...) live at the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
