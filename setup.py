from pathlib import Path

from setuptools import setup

setup(version=Path("rl_zoo3/version.txt").read_text(encoding="utf-8").strip())
