"""
Compatibility shim for STCQA runner.
Redirects to src.models.stcqa.run.
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.stcqa.run import main

if __name__ == "__main__":
    main()
