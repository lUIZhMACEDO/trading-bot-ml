"""Start the browser dashboard without installing the package: ``python dashboard.py``.

Same as ``trading-bot-dashboard`` after ``pip install -e .``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from trading_bot.dashboard.app import main  # noqa: E402

if __name__ == "__main__":
    main()
