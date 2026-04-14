#!/usr/bin/env python3
"""
QUANTUM TRADER — One-Click Setup & Launcher
=============================================
Just run: python setup_and_run.py

This script:
  1. Checks your Python version
  2. Installs all required packages
  3. Launches the trading bot
"""

import subprocess
import sys
import os

def main():
    print()
    print("  ╔═══════════════════════════════════════════════╗")
    print("  ║   ⚛  QUANTUM TRADER v2.0 — Setup & Launch    ║")
    print("  ╚═══════════════════════════════════════════════╝")
    print()

    # ── Step 1: Check Python version ──
    v = sys.version_info
    print(f"  [1/3] Python version: {v.major}.{v.minor}.{v.micro}", end="")
    if v.major < 3 or (v.major == 3 and v.minor < 9):
        print(" ❌")
        print("        Need Python 3.9+. Download from https://python.org")
        sys.exit(1)
    print(" ✅")

    # ── Step 2: Install packages ──
    packages = [
        "alpaca-py",
        "python-dotenv",
        "pandas",
        "numpy",
        "scikit-learn",
        "yfinance",
        "ta",
        "fastapi",
        "uvicorn[standard]",
    ]

    # Optional but recommended
    optional = [
        "lightgbm",
        "fredapi",
    ]

    print(f"  [2/3] Installing {len(packages)} required packages...")
    print()

    for pkg in packages:
        print(f"        Installing {pkg}...", end=" ", flush=True)
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", pkg, "-q", "--disable-pip-version-check"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print("✅")
        else:
            # Try with --break-system-packages for Linux
            result2 = subprocess.run(
                [sys.executable, "-m", "pip", "install", pkg, "-q",
                 "--disable-pip-version-check", "--break-system-packages"],
                capture_output=True, text=True
            )
            if result2.returncode == 0:
                print("✅")
            else:
                print("⚠️  (may need manual install)")

    print()
    print(f"        Installing optional packages (LightGBM, FRED)...")
    for pkg in optional:
        print(f"        Installing {pkg}...", end=" ", flush=True)
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", pkg, "-q", "--disable-pip-version-check"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print("✅")
        else:
            print("⏭  (skipped — bot works without it)")

    # ── Step 3: Launch the bot ──
    print()
    print(f"  [3/3] Launching Quantum Trader...")
    print()
    print("  " + "─" * 50)
    print()

    # Find the bot script
    bot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quantum_web.py")
    if not os.path.exists(bot_path):
        bot_path = "quantum_web.py"
        if not os.path.exists(bot_path):
            # Fall back to terminal version
            bot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quantum_trader_v2.py")
            if not os.path.exists(bot_path):
                bot_path = "quantum_trader_v2.py"
            if not os.path.exists(bot_path):
                print("  ❌ Can't find quantum_web.py or quantum_trader_v2.py")
                print("     Make sure they're in the same folder as this script.")
                sys.exit(1)

    os.execv(sys.executable, [sys.executable, bot_path])


if __name__ == "__main__":
    main()
