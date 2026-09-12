#!/usr/bin/env python3
"""Thin command entry point for the installed SEVC source distribution."""
from pathlib import Path
import sys

# Prefer the project package over this identically named entry-point file.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sevc.experiments.source_distribution import main

if __name__ == "__main__":
    raise SystemExit(main())
