import json
import pathlib
import os
import pandas as pd
import matplotlib.pyplot as plt
import pytest
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


def json_to_dataframe(filepath):
    """Loads a JSON file where the benchmark is stored and returns a dataframe
    with the benchmar information."""
    with open(filepath) as f:
        data = json.load(f)
        cpu = data['machine_info']["cpu"]["brand_raw"]
        time = data['commit_info']["time"]
        
        data = data['benchmarks']
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
        cpu = cpu.replace("@","at")
        cpu = cpu.replace(".","-")
        cpu = cpu.replace("(R)", "")
        
        time = time[:-6]
        time = time.replace("T", " ")
        data['cpu']= cpu
        data['time'] = time
        
        return data


def plot_benchmark(df):
    """Plots results using matplotlib. It iterates params_get_operation and
    params_density and plots time vs N (for NxN matrices)"""
    grouped = df.groupby(['params_get_operation', 'params_density','params_size','cpu'])
    for (operation, density, size, cpu), group in grouped:
        for dtype, g in group.groupby('extra_info_dtype'):
            plt.errorbar(g.time, g.stats_mean, g.stats_stddev,
                         fmt='.-', label=dtype)

        plt.title(f"{operation} {density} {size} {cpu}")
        plt.legend()        
        plt.xlabel("date")
        plt.xticks(rotation=90)
        plt.yscale('log')
        plt.ylabel("time (s)")
        plt.tight_layout()
        plt.savefig(f"./.benchmarks/figures/{cpu}_{operation}_{density}_{size}.png")
        plt.close()

def get_paths():
    """Returns the path to the latest benchmark run from `./.benchmarks/`"""

    benchmark_paths = glob.glob("./.benchmarks/*/*.json")
    

    return benchmark_paths

def create_dataframe(paths):
    df = pd.DataFrame()
 
    for path in paths:
        data = json_to_dataframe(path)
        df = pd.concat([df,data])
    
    return df



def main(args=[]):
    folder = ".benchmarks/figures"
    paths = get_paths()
    data = create_dataframe(paths)
    Path(folder).mkdir(parents=True, exist_ok=True)
    plot_benchmark(data,)

  
if __name__ == '__main__':
    main()