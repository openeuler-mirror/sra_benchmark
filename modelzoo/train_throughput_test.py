import argparse
import json
import os
import re
import subprocess
import time
from itertools import product
from multiprocessing import Process, Queue


def get_numa_info():
    # Run the 'lscpu' command to get the CPU and NUMA node information
    result = subprocess.run(['lscpu'], stdout=subprocess.PIPE, text=True)
    output = result.stdout

    # NUMA 节点0 CPU：      0-79
    pattern_numa = re.compile(r'NUMA node\d CPU\(s\):\s*(.*)')
    pattern_cpu =  re.compile(r"CPU$s$:\s*(\d+)")

    matches_cpu = pattern_cpu.findall(output)
    matches_numa = pattern_numa.findall(output)
    cpu_info = ''
    if matches_cpu:
        cpu_info = matches_cpu[0]

    numa_info = {}
    if matches_numa:
        for numa_value in matches_numa:
            numa_node = f'node{len(numa_info)}'
            numa_info[numa_node] = numa_value
    return cpu_info, numa_info


def generate_train_cmd(server_numa, model_name, meta_path, data_path):
    train_cmd = f"taskset -c {server_numa} python {meta_path}/modelzoo/{model_name}/train.py --tf --data_location {data_path} --output_dir {os.path.join(meta_path, 'modelzoo', model_name, 'result')}"
    print(train_cmd)
    return train_cmd


def run_command(cmd):
    """Run a shell command and return its output."""
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return stdout.decode('utf-8'), stderr.decode('utf-8')


def parse_perf_analyzer_output(meta_path, model_name):
    """Parse the perf_analyzer output to extract throughput."""
    throughput = 0
    with open(os.path.join(meta_path, "modelzoo", f"{model_name}", "result", f"{model_name}_throughput_record.txt"), "r") as f:
        for line in f.readlines():
            if model_name in line:
                print(line)
                res = line.strip().split()
                throughput = float(res[-1])
                break

    return throughput


def run_train(test_method, meta_path, criteo_data_path, taobao_data_path):
    cpu_info, numa_info = get_numa_info()
    server_numa = numa_info["node0"]
    print(f"server_numa: {server_numa}")

    model_list = ["wide_and_deep", "dlrm", "deepfm", "dffm", "dssm", "esmm"]

    if test_method == "entire":
        server_numa = numa_info["node0"]
    else:
        server_numa = "0-" + str(int(cpu_info) - 1)

    for i in range(len(model_list)):
        if model_list[i] == "dssm" or model_list[i] == "esmm":
            train_cmd = generate_train_cmd(server_numa, model_list[i], meta_path, taobao_data_path)
        else:
            train_cmd = generate_train_cmd(server_numa, model_list[i], meta_path, criteo_data_path)
        stdout, stderr = run_command(train_cmd)
        if stderr:
            print(model_list[i] + " 训练未完成")
            print(stderr)
        else:
            print(model_list[i] + " 训练完成")
            print(stdout)

    throughputs = {}
    for model_name in model_list:
        throughput = parse_perf_analyzer_output(meta_path, model_name)
        throughputs[model_name] = throughput

    with open(os.path.join(meta_path, "modelzoo", "train_throughput.txt"), "w") as file:
        json.dump(throughputs, file, indent=4)


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_method",
                        help="single numa or entire machine",
                        type=str,
                        default="single")
    parser.add_argument("--meta_path",
                        help="full path of modelzoo",
                        type=str,
                        default="/home/r00813794")
    parser.add_argument("--criteo_data_location",
                        help="set the path of tritonclient",
                        type=str,
                        default="/home/r00813794/modelzoo/wide_and_deep/data")
    parser.add_argument("--taobao_data_location",
                        help="set the path of tritonclient",
                        type=str,
                        default="/home/r00813794/modelzoo/dssm/data")

    return parser


if "__main__" == __name__:
    parser = get_arg_parser()
    args = parser.parse_args()
    run_train(args.test_method, args.meta_path, args.criteo_data_location, args.taobao_data_location)
