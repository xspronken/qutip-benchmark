import json
import pathlib
import os
import pandas as pd
import matplotlib.pyplot as plt
import pytest
import argparse
import glob

from importlib.metadata import version

from pathlib import Path



def get_latest_benchmark_path():
    """Returns the path to the latest benchmark run from `./.benchmarks/`"""

    benchmark_paths = glob.glob("./.benchmarks/*/*.json")
    dates = [''.join(_b.split("/")[-1].split('_')[2:4])
             for _b in benchmark_paths]
    benchmarks = {date: value for date, value in zip(dates, benchmark_paths)}

    dates.sort()
    latest = dates[-1]
    benchmark_latest = benchmarks[latest]

    return benchmark_latest

def add_packages_to_json(filepath):
    """Loads a JSON file where the benchmark is stored and returns a dataframe
    with the benchmar information."""
    with open(filepath,"r") as f:
        data = json.load(f)
        data['package_versions'] = {}
        data['package_versions']['scipy'] = version('scipy')
        data['package_versions']['numpy'] = version('numpy')
        data['package_versions']['cython'] = version('cython')
        data['package_versions']['qutip'] = version('qutip')
        data['package_versions']['pytest'] = version('pytest')
        data['package_versions']['pytest-benchmark'] = version('pytest-benchmark')
    with open(filepath,"w") as f: 
        json.dump(data,f, indent=4, separators=(',', ': '))
      


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
    path = get_latest_benchmark_path()
    add_packages_to_json(path)


if __name__ == '__main__':
    main()
