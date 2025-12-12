"""Test configuration.

Ensures `import axm` works when running tests without installing the package.
"""

from __future__ import annotations

import sys
from pathlib import Path


SRC = (Path(__file__).resolve().parents[1] / "src").as_posix()
if SRC not in sys.path:
    sys.path.insert(0, SRC)
