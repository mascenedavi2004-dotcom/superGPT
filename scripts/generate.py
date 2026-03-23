#!/usr/bin/env python3
"""Generate text from a superGPT model."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import runpy
runpy.run_module("supergpt.inference.generate", run_name="__main__")
