from pathlib import Path

__version__ = Path(__file__).parent.joinpath("version.txt").read_text(encoding="utf-8").strip()
