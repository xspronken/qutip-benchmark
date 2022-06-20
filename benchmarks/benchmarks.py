import json
import pathlib
import os
import pandas as pd
import matplotlib.pyplot as plt
import pytest
import argparse
import glob
from pathlib import Path






def run_benchmarks(args):
    "Run pytest benchmark with sensible defaults."
    pytest.main(["--benchmark-only",
                 "--benchmark-columns=Mean,StdDev,rounds,Iterations",
                 "--benchmark-sort=name",
                 "--benchmark-autosave",
                 "-Wdefault"] +
                args)



def main(args=[]):
    parser = argparse.ArgumentParser(description="""Run and plot the benchmarks.
                                     The script also accepts the same arguments
                                     as pytest/pytest-benchmark. The script must be run
                                     from the root of the repository.""")
    run_benchmarks(args)


if __name__ == '__main__':
    main()
