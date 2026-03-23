#!/usr/bin/env python3
"""Fine-tune a superGPT model."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import runpy
runpy.run_module("supergpt.training.finetune", run_name="__main__")
