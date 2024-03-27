from __future__ import annotations

from sybil import Sybil
from sybil.parsers.markdown import PythonCodeBlockParser

# Enable sybil to collect ```python codeblocks in docstrings
pytest_collect_file = Sybil(
    parsers=[
        PythonCodeBlockParser(),
    ],
    patterns=["*.py"],
).pytest()
