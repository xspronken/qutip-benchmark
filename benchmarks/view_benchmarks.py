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

def compute_stats(df):
    column_names = ["title",'dtype','count',"mean",'stddev','min','5%','q1','median','q3','95%','max']
    data = []
    stats = pd.DataFrame(columns=column_names)
    grouped = df.groupby(['params_get_operation', 'params_density','params_size','cpu'])
    for (operation, density, size, cpu), group in grouped:
        for dtype, g in group.groupby('extra_info_dtype'):
            data.append([f"{operation} {density} {size} {cpu}", dtype,
            g.stats_mean.describe()['count'], g.stats_mean.describe()['mean'],
            g.stats_mean.describe()['std'], g.stats_mean.describe()['min'],
            g.stats_mean.describe(percentiles=[0.05])['5%'], g.stats_mean.describe()['25%'],
            g.stats_mean.describe()['50%'], g.stats_mean.describe()['75%'],
            g.stats_mean.describe(percentiles=[0.95])['95%'], g.stats_mean.describe()['max']])

    datadf = pd.DataFrame(data, columns= column_names)
    return datadf

def plot_benchmark_dtype(df):
    """Plots results using matplotlib. It iterates params_get_operation and
    params_density and plots time vs N (for NxN matrices)"""
    grouped = df.groupby(['params_get_operation','params_density','params_size'])
    for (operation, density, size), group in grouped:
        if size > 200  and operation == 'matmul':
            N = len(group.groupby('extra_info_dtype'))
            fig, ax = plt.subplots(N,2,sharex=True,gridspec_kw={'width_ratios': [3, 1]})
            fig.set_size_inches(15, 6*N)
            fig.suptitle(f"{operation} {density} {size}")
            n=0
            for dtype, g in group.groupby('extra_info_dtype'):
  
                x = g.stats_mean.describe(percentiles=[0.05,0.25,0.5,0.75,0.95])
                median = [x['50%'] for i in g.stats_median]
                medianplus10 = [i*1.1 for i in median]
                medianminus10 = [i*0.9 for i in median]

                
                ax[n,0].errorbar(g.time, g.stats_mean, g.stats_stddev,
                                fmt='.-', label=dtype)
                
                ax[n,0].plot(g.time,median, label='median')
                ax[n,0].plot(g.time,medianplus10, label='median + 10%')
                ax[n,0].plot(g.time,medianminus10, label='median - 10%')
                
                
                
                ax[n,1].set_axis_off()
                ax[n,1].text(0,0.5,f"{x}")
                ax[n,0].legend()        
                ax[n,0].set_xlabel("date")
                ax[n,0].tick_params(labelrotation=90)
                ax[n,0].set_ylabel("time (s)")
                n = n+1
            plt.savefig(f"./.benchmarks/figures/dtype_sep/{operation}_{density}_{size}.png",bbox_inches='tight')
            plt.close()

def plot_benchmark_cpu_dtype(df):
    """Plots results using matplotlib. It iterates params_get_operation and
    params_density and plots time vs N (for NxN matrices)"""
    grouped = df.groupby(['params_get_operation','params_density','params_size','cpu'])
    for (operation, density, size, cpu), group in grouped:
        if size > 200  and operation == 'matmul':
            N = len(group.groupby('extra_info_dtype'))
            fig, ax = plt.subplots(N,2,sharex=True,gridspec_kw={'width_ratios': [3, 1]})
            fig.set_size_inches(15, 6*N)
            fig.suptitle(f"{operation} {density} {size} {cpu}")
            n=0
            for dtype, g in group.groupby('extra_info_dtype'):
                    x = g.stats_mean.describe(percentiles=[0.05,0.25,0.5,0.75,0.95])
                    median = [x['50%'] for i in g.stats_median]
                    medianplus10 = [i*1.1 for i in median]
                    medianminus10 = [i*0.9 for i in median]

                    
                    ax[n,0].errorbar(g.time, g.stats_mean, g.stats_stddev,
                                    fmt='.-', label=dtype)
                    
                    ax[n,0].plot(g.time,median, label='median')
                    ax[n,0].plot(g.time,medianplus10, label='median + 10%')
                    ax[n,0].plot(g.time,medianminus10, label='median - 10%')
                    
                    
                    
                    ax[n,1].set_axis_off()
                    ax[n,1].text(0,0.5,f"{x}")
                    ax[n,0].legend()        
                    ax[n,0].set_xlabel("date")
                    ax[n,0].tick_params(labelrotation=90)
                    ax[n,0].set_ylabel("time (s)")
                    n = n+1
            plt.savefig(f"./.benchmarks/figures/cpu_dtype_sep/{cpu}_{operation}_{density}_{size}.png",bbox_inches='tight')
            plt.close()

def plot_benchmark_cpu(df):
    """Plots results using matplotlib. It iterates params_get_operation and
    params_density and plots time vs N (for NxN matrices)"""
    grouped = df.groupby(['params_get_operation','params_density','params_size','cpu'])
    for (operation, density, size, cpu), group in grouped:
        if size > 200 and operation == 'matmul':
            d = {}
            fig, ax = plt.subplots(1,1,sharex=True)
            # fig.set_size_inches(15, 15)
            fig.suptitle(f"{operation} {density} {size} {cpu}")
            for dtype, g in group.groupby('extra_info_dtype'):
                ax.errorbar(g.time, g.stats_mean, g.stats_stddev,
                            fmt='.-', label=dtype)
                d[dtype]=g.stats_mean
            ax.legend()        
            ax.set_xlabel("date")
            ax.tick_params(labelrotation=90)
            ax.set_ylabel("time (s)")
            ax.set_yscale('log')
            # list1 = [i/j for i,j in zip(d['numpy'],d["qutip_dense"])]

            # list2 = [i/j for i,j in zip(d['scipy_csr'],d["qutip_csr"])]

            # ax[1].plot(list1, label= "numpy/qutip_dense")
 
            # ax[1].plot(list2, label= "scipy_csr/qutip_sparse")
  
            # ax[1].legend()
            # ax[1].set_xlabel("date")
            # ax[1].tick_params(labelrotation=90)
            # ax[1].set_ylabel("time (s)")

            plt.tight_layout()
            plt.savefig(f"./.benchmarks/figures/cpu_sep/{cpu}_{operation}_{density}_{size}.png")
            plt.close()

def plot_benchmark(df):
    """Plots results using matplotlib. It iterates params_get_operation and
    params_density and plots time vs N (for NxN matrices)"""
    grouped = df.groupby(['params_get_operation','params_density','params_size'])
    for (operation, density, size), group in grouped:
        if size > 200 and operation == 'matmul':
            for dtype, g in group.groupby('extra_info_dtype'):
                plt.errorbar(g.time, g.stats_mean, g.stats_stddev,
                            fmt='.-', label=dtype)
            
            plt.title(f"{operation} {density} {size}")
            plt.legend()        
            plt.xlabel("date")
            plt.xticks(rotation=90)
            plt.ylabel("time (s)")
            plt.tight_layout()
            plt.savefig(f"./.benchmarks/figures/no_sep/{operation}_{density}_{size}.png")
            plt.close()

def count_cpu(df):
    grouped = df.groupby('time')
    cpus = []
    res = {}
    for time, g in grouped:
        cpus.append(g.cpu[0])
    n = len(cpus)
    for i in cpus:
        res[i] = [cpus.count(i), cpus.count(i)/n]
    print(res)

def get_paths():
    """Returns the path to the latest benchmark run from `./.benchmarks/`"""

    benchmark_paths = glob.glob("./.benchmarks/*/*.json")
    dates = [''.join(_b.split("/")[-1].split('_')[2:4])
             for _b in benchmark_paths]
    zipped = zip(dates, benchmark_paths)
    tmp = sorted(zipped, key = lambda x: x[0])
    res = list(zip(*tmp))
    
    return res[1]

def create_dataframe(paths):
    df = pd.DataFrame()

    for path in paths:
        
        data = json_to_dataframe(path)
        df = pd.concat([df,data])
    
    return df



def main(args=[]):
    folder = Path(".benchmarks/figures")
    paths = get_paths()
    data = create_dataframe(paths)
    folder.mkdir(parents=True, exist_ok=True)
    count_cpu(data)
    plot_benchmark(data)
    print('no sep done')
    plot_benchmark_dtype(data)
    print('dtype sep done')
    plot_benchmark_cpu(data)
    print('cpu sep done')
    plot_benchmark_cpu_dtype(data)
    print('dtype cpu sep done')
 

  
if __name__ == '__main__':
    main()