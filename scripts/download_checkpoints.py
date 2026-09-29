#!/usr/bin/env python3
"""
Wrapper forwarding to src.utils.download_checkpoints.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.download_checkpoints import main

if __name__ == "__main__":
    main()
