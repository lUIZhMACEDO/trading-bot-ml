"""Run the terminal interface without installing the package: ``python cli.py``.

Same as ``python -m trading_bot`` after ``pip install -e .``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from trading_bot.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
