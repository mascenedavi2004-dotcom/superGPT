#!/usr/bin/env python3
"""Export model to GGUF format."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import runpy
runpy.run_module("supergpt.inference.export", run_name="__main__")
