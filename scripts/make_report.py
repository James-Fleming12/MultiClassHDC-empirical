#!/usr/bin/env python
"""Build all tables and figures from the raw run records."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hdc_bench.aggregate import main

if __name__ == "__main__":
    main()
