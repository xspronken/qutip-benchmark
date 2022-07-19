import json
from lib2to3.pgen2.pgen import DFAState
import pathlib
import os
from re import M

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import pytest
import argparse
import glob
from pathlib import Path
from datetime import datetime


def unravel(data, key):
    """Transforms {key:{another_key: values, another_key2: value2}} into
    {key_another_key:value, key_another_key2:value}"""
    for d in data:
        values = d.pop(key)
        for k, v in values.items():
            d[key+'_'+k] = v
    return data


def remove_outliers(df,col,n_std):  
    mean = df[col].mean()
    sd = df[col].std()
    
    df = df[(df[col] <= mean+(n_std*sd))]
    return df
def delete_date_duplicates(df):
    df["date"] = df["time"].dt.date
    df.drop_duplicates(subset=["date"], keep='last')
    print(df.date)
    return df

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
        data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')

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
        if size == 128 or size == 512:

            fig, ax = plt.subplots(1,1)
            fig.suptitle(f'Matrix density: {density}         Matrix Size: {size}x{size}', fontsize=20)
            fig.set_size_inches(9, 9)
            for dtype, g in group.groupby('extra_info_dtype'):
                colors = ["blue", "orange", "green", "red"]
                markers = ['o--','x-','v:']
                count = 0
                cpus = []
                for cpu, gra in g.groupby('cpu'):
                    gr = delete_date_duplicates(gra)
                    if "Platinum" in cpu:
                        cpus.append(cpu)
                        if dtype == 'numpy':
                            ax.plot(gr.time, gr.stats_mean, markers[count], color=colors[0])
                        if dtype == 'qutip_dense':
                            ax.plot(gr.time, gr.stats_mean, markers[count], color=colors[1])
                        if dtype == 'qutip_csr':
                            ax.plot(gr.time, gr.stats_mean, markers[count], color=colors[2])
                        if dtype == 'scipy_csr':
                            ax.plot(gr.time, gr.stats_mean, markers[count], color=colors[3])
                        count = count+1                  
                f = lambda m,c: plt.plot([],[],m, color=c)[0]

            handles = [f("s", colors[i]) for i in range(4)]
            handles += [f(markers[i], "k") for i in range(3)]
            labels = ['numpy','qutip_dense','qutip_csr','scipy_csr'] + cpus
            ax.legend(handles, labels) 
            ax.set_xlabel("date")
            ax.tick_params(labelrotation=90)
            ax.set_ylabel("time (s)")
            ax.set_yscale('log')
            fig.tight_layout()
            fig.subplots_adjust(top=0.95)
            


                
            plt.gcf().autofmt_xdate()

            folder = Path("images/plots")
            folder.mkdir(parents=True, exist_ok=True)
            plt.savefig(f"./images/plots/{operation}_{density}_{size}.png",bbox_inches='tight')
            plt.close()

def plot_benchmark_cpu_dtype(df):
    """Plots results using matplotlib. It iterates params_get_operation and
    params_density and plots time vs N (for NxN matrices)"""
    folder = Path("images/cpu_dtype_sep")
    folder.mkdir(parents=True, exist_ok=True)
    grouped = df.groupby(['params_get_operation','params_density','params_size','cpu'])
    for (operation, density, size, cpu), group in grouped:
        if size > 200  and operation == 'matmul':
            N = len(group.groupby('extra_info_dtype'))
            fig, ax = plt.subplots(N,1)
            fig.set_size_inches(15, 6*N)
            fig.suptitle(f"{operation} {density} {size} {cpu}")
            n=0
            for dtype, g in group.groupby('extra_info_dtype'):
                    x = g.stats_mean.describe(percentiles=[0.05,0.25,0.5,0.75,0.95])
                    median = [x['50%'] for i in g.stats_median]
                    medianplus10 = [i*1.1 for i in median]
                    medianminus10 = [i*0.9 for i in median]

                    
                    ax[n].errorbar(g.time, g.stats_mean, g.stats_stddev,
                                    fmt='.-', label=dtype)
                    
                    ax[n].plot(g.time,median, label='median')
                    ax[n].plot(g.time,medianplus10, label='median + 10%')
                    ax[n].plot(g.time,medianminus10, label='median - 10%')
                    ax[n].legend()        
                    ax[n].set_xlabel("date")
                    ax[n].tick_params(labelrotation=90)
                    ax[n].set_ylabel("time (s)")
                    n = n+1
            plt.gcf().autofmt_xdate()
            plt.savefig(f"./images/cpu_dtype_sep/{cpu}_{operation}_{density}_{size}.png",bbox_inches='tight')
            plt.close()

def plot_benchmark_cpu(df):
    """Plots results using matplotlib. It iterates params_get_operation and
    params_density and plots time vs N (for NxN matrices)"""
    folder = Path("images/cpu_sep")
    folder.mkdir(parents=True, exist_ok=True)
    grouped = df.groupby(['params_get_operation','params_density','params_size','cpu'])
    for (operation, density, size, cpu), group in grouped:
        if size > 200 and operation == 'matmul':
            d = {}
            fig, ax = plt.subplots(1,1)
            fig.set_size_inches(15, 15)
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


            plt.tight_layout()
            plt.gcf().autofmt_xdate()
            plt.savefig(f"./images/cpu_sep/{cpu}_{operation}_{density}_{size}.png")
            plt.close()

def plot_benchmark(df):
    """Plots results using matplotlib. It iterates params_get_operation and
    params_density and plots time vs N (for NxN matrices)"""
    folder = Path("images/no_sep")
    folder.mkdir(parents=True, exist_ok=True)
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
            plt.gcf().autofmt_xdate()
            plt.savefig(f"./images/no_sep/{operation}_{density}_{size}.png")
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
    folder = Path("images")
    paths = get_paths()
    data = create_dataframe(paths)
    folder.mkdir(parents=True, exist_ok=True)

    # plot_benchmark(data)
    # print('no sep done')
   # plot_benchmark_dtype(data)
    plot_benchmark_dtype(data)
    print('done')
    # plot_benchmark_cpu(data)
    # print('cpu sep done')
    # plot_benchmark_cpu_dtype(data)
    # print('dtype cpu sep done')
 

  
if __name__ == '__main__':
    main()