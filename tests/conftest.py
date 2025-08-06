import sys
from pathlib import Path

# Adds project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
