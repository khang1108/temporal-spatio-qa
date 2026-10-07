"""Entrypoint proxy for scripts/datasets/run_eval.py
"""
import os
import sys

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.datasets.run_eval import main

if __name__ == "__main__":
    main()

