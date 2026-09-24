#!/usr/bin/env python3
"""
MoneyPrinterTurbo Unified Local Development Launcher.
Orchestrates FastAPI Backend, StageWorker, and Streamlit WebUI as independent OS processes.

Usage:
    python dev.py                  # Start full environment (Backend + Worker + WebUI)
    python dev.py --check-only     # Run pre-flight checks and exit
    python dev.py --no-webui       # Start Backend + Worker without WebUI
    python dev.py --no-worker      # Start Backend + WebUI without Worker
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.dev_launcher import main

if __name__ == "__main__":
    main()
