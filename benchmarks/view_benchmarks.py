import json
import pathlib
import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import glob
from pathlib import Path


def unravel(data, key):
    """Transforms {key:{another_key: values, another_key2: value2}} into
    {key_another_key:value, key_another_key2:value}"""
    for d in data:
        values = d.pop(key)
        for k, v in values.items():
            d[key+'_'+k] = v
    return data


def benchmark_to_dataframe(filepath):
    """Loads a JSON file where the benchmark is stored and returns a dataframe
    with the benchmar information."""
    with open(filepath) as f:
        data = json.load(f)
        info , data = data['machine_info'], data['benchmarks']
  
        data = unravel(data, 'options')
        data = unravel(data, 'stats')
        data = unravel(data, 'params')
        data = unravel(data, 'extra_info')
        
       
        
        
        data = pd.DataFrame(data)
        

        # Set operation properly (for example: matmul instead of:
        # UNSERIALIZABLE[<function Qobj.__matmul__ at 0x...)
        # The name of the operation is obtained from the group name
        data.params_get_operation = data.group.str.split('-')
        data.params_get_operation = [d[-1] for d in data.params_get_operation]

        return data , info 


def plot_benchmark(df, destination_folder):
    """Plots results using matplotlib. It iterates params_get_operation and
    params_density and plots time vs N (for NxN matrices)"""
    grouped = df.groupby(['params_get_operation', 'params_density'])
    for (operation, density), group in grouped:
        for dtype, g in group.groupby('extra_info_dtype'):
            plt.errorbar(g.params_size, g.stats_mean, g.stats_stddev,
                         fmt='.-', label=dtype)

        plt.title(f"{operation} {density}")
        plt.legend()
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel("N")
        plt.ylabel("time (s)")
        plt.savefig(f".benchmarks/figures/{operation}_{density}.png")
        plt.close()


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


def main():
    benchmark_latest = get_latest_benchmark_path()
    _ , info  = benchmark_to_dataframe(benchmark_latest)

    print(info["node"])


if __name__ == '__main__':
    main()
