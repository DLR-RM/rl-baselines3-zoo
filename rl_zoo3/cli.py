import sys

from rl_zoo3.enjoy import enjoy
from rl_zoo3.train import train


def main():
    script_name = sys.argv[1]
    # Remove script name
    del sys.argv[1]
    # Execute known script
    {
        "train": train,
        "enjoy": enjoy,
    }[script_name]()
