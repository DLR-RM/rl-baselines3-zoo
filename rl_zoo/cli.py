import sys

from rl_zoo.enjoy import enjoy
from rl_zoo.train import train


def main():
    script_name = sys.argv[1]
    # Remove script name
    del sys.argv[1]
    # Execute known script
    {
        "train": train,
        "enjoy": enjoy,
    }[script_name]()
