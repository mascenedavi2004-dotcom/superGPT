#!/usr/bin/env python3
"""Knowledge distillation for superGPT."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import runpy
runpy.run_module("supergpt.training.distill", run_name="__main__")
