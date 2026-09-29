#!/usr/bin/env python3
"""
Wrapper forwarding to src.evaluation.benchmark.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.evaluation.benchmark import main

if __name__ == "__main__":
    main()
