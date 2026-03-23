#!/usr/bin/env python3
"""RLVR — RL with Verifiable Rewards (DeepSeek R1 style)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import runpy
runpy.run_module("supergpt.alignment.rlvr", run_name="__main__")
