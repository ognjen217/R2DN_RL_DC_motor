#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import shutil
import sys


TARGET = Path("src/r2dn_dc_motor/validation/thermal_test_bank.py")
HELPER_NAME = "_summarize_full_rk4_result"
MARKER = "FULL/RK4 diagnostic:"


HELPER = r