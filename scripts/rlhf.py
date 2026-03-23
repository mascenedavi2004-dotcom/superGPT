#!/usr/bin/env python3
"""RLHF alignment (PPO, GRPO, DAPO) for superGPT."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import runpy
runpy.run_module("supergpt.alignment.rlhf", run_name="__main__")
