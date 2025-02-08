import argparse
import os
import re
import subprocess
import time
from multiprocessing import Process


def get_numa_info():
    # Run the 'lscpu' command to get the CPU and NUMA node information
    result = subprocess.run(['lscpu'], stdout=subprocess.PIPE, text=True)
    output = result.stdout

    # NUMA 节点0 CPU：      0-79
    pattern = re.compile(r'NUMA node\d CPU\(s\):\s*(.*)')
    numa_info = {}
    # Find all matches in the lscpu output
    for match in pattern.finditer(output):
        cpus = match.group(1)
        # Store the CPU list in a dictionary with the corresponding NUMA node
        numa_node = f'node{len(numa_info)}'
        numa_info[numa_node] = cpus

    return numa_info


def start_container(image, name, numa, command=None):
    """启动 Docker 容器"""
    if command:
        run_command(f"docker run -d --rm --name {name} --cpuset-cpus='{numa}' --net host {image} {command}")
    else:
        run_command(f"docker run -d --rm --name {name} --cpuset-cpus='{numa}' --net host {image}")


def exec_command_in_container(container_name, command):
    """在指定的 Docker 容器中执行命令并返回输出"""
    stdout, stderr = run_command(f"docker exec {container_name} {command}")
    return stdout, stderr


def run_command(cmd):
    """Run a shell command and return its output."""
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return stdout.decode('utf-8'), stderr.decode('utf-8')


def stop_and_remove_container(container_name):
    """停止并删除 Docker 容器"""
    run_command(f"docker stop {container_name}")
    run_command(f"docker rm {container_name}")


def start_server(serving_id, server_memory, server_numa, serving_path, port, model_path, model_name, intra, inter,
                 min_cluster_size=200, enable_xla=False, enable_onednn=False):
    """Start the TensorFlow server with the specified parameters."""
    my_env = os.environ.copy()

    if enable_onednn:
        # enable_onednn_cmds = """export TF_ENABLE_ONEDNN_OPTS=1"""
        my_env['TF_ENABLE_ONEDNN_OPTS'] = "1"
        print("enable oneDNN")
    else:
        my_env['TF_ENABLE_ONEDNN_OPTS'] = "0"
        print("disable oneDNN")

    if enable_xla:
        my_env["TF_XLA_FLAGS"] = f"--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit --tf_xla_enable_lazy_compilation=false --tf_mlir_enable_merge_control_flow_pass  --tf_xla_min_cluster_size={min_cluster_size} --tf_xla_disable_full_embedding_pipelining"
        my_env["XLA_LLVM_UNROLL"] = "true"
        print("enable XLA")
        server_cmd = f"nohup numactl -m {server_memory} -C {server_numa} {serving_path} --port={port} --model_base_path={model_path} --model_name={model_name} --tensorflow_intra_op_parallelism={intra} --tensorflow_inter_op_parallelism={inter} --xla_cpu_compilation_enabled=true > output_{model_name}_{serving_id}.log 2>&1 &"
    else:
        try:
            my_env.pop("TF_XLA_FLAGS")
            my_env.pop("XLA_LLVM_UNROLL")
        except KeyError:
            print("disable XLA")
        server_cmd = f"nohup numactl -m {server_memory} -C {server_numa} {serving_path} --port={port} --model_base_path={model_path} --model_name={model_name} --tensorflow_intra_op_parallelism={intra} --tensorflow_inter_op_parallelism={inter} --xla_cpu_compilation_enabled=false > output_{model_name}_{serving_id}.log 2>&1 &"

    print("server_cmd")
    print(server_cmd)
    result1 = subprocess.run("env | grep TF_ENABLE_ONEDNN_OPTS", shell=True, env=my_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    result2 = subprocess.run("env | grep TF_XLA_FLAGS", shell=True, env=my_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(result1.stdout)
    print(result2.stdout)
    process = subprocess.Popen(server_cmd, shell=True, env=my_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print(stdout)
    print(stderr)


def start_pressure_test(meta_path, container):
    """Start the triton with the specified parameters."""
    image = container["image"]
    name = container["name"]
    numa = container["numa"]
    model_name = container["model_name"]
    exec_command = container["exec_command"]

    # 启动容器
    start_container(image, name, numa, "sleep infinity")
    print(f"Container {container['name']} started")

    # 等待一段时间，确保容器已经启动
    time.sleep(10)

    # 在单个容器中执行命令并将输出保存到文件
    stdout, stderr = exec_command_in_container(name, exec_command)
    output_file = os.path.join(meta_path, "modelzoo", "inference_log")
    if not os.path.exists(output_file):
        os.mkdir(output_file)
    output_filename = os.path.join(output_file, f"{model_name}_{name}.txt")
    save_to_file(output_filename, stdout)
    print(f"Output from {name} for command '{exec_command}' saved to {output_filename}")
    if stderr:
        error_filename = os.path.join(output_file, f"{model_name}_{name}_error.txt")
        save_to_file(error_filename, stderr)
        print(f"Error from {name} for command '{exec_command}' saved to {error_filename}")


def save_to_file(filename, content):
    with open(filename, "w+") as file:
        file.write(content + "\r\n")


def start_server_and_client(meta_path: str, concurrency: str, batch: int, client_numa: list, server_numa: list,
                            measurement_interval: int, model_name: str, model_path: str, serving_path: str, ports: list,
                            intra: int, inter: int, min_cluster_size: int, enable_xla: bool, enable_onednn: bool,
                            image: str = "nvcr.io/nvidia/tritonserver:24.05-py3-sdk"):
    subprocess.run(f"pkill -9 tensorflow_mode", shell=True)  # Kill previous instance

    # Start the TensorFlow server in a separate process
    server_process_list = []
    for i in range(len(server_numa)):
        server_process = Process(target=start_server,
                                 args=(
                                     i, i, server_numa[i], serving_path, ports[i], model_path, model_name, intra, inter,
                                     min_cluster_size, enable_xla, enable_onednn))
        server_process.start()
        server_process_list.append(server_process)

    time.sleep(5)

    request_process_list = []
    for i in range(len(server_numa)):
        container = {"image": image,
                     "name": "client" + str(i),
                     "numa": client_numa[i],
                     "model_name": model_name,
                     "exec_command": f"perf_analyzer --concurrency-range {concurrency} -p {measurement_interval} --latency-threshold 200 -f perf.csv -m {model_name} --service-kind tfserving -i grpc --request-distribution poisson -b {batch} -u localhost:{ports[i]} --percentile 99 --input-data=random"}
        request_process = Process(target=start_pressure_test, args=(meta_path, container,))
        request_process.start()
        request_process_list.append(request_process)
    concurrency_list = concurrency.split(":")
    if len(concurrency_list) == 3:
        num_concurrency = (int(concurrency_list[1]) - int(concurrency_list[0])) // int(concurrency_list[2]) + 1
    else:
        num_concurrency = (int(concurrency_list[1]) - int(concurrency_list[0])) // 1 + 1
    sleep_time = num_concurrency * measurement_interval // 1000 * 5
    time.sleep(sleep_time)

    for p in server_process_list:
        p.terminate()
        p.join()
    for p in request_process_list:
        p.terminate()
        p.join()

    for i in range(len(server_numa)):
        container_name = "client" + str(i)
        stop_and_remove_container(container_name)


def parse_perf_analyzer_output(meta_path, model_name, num_numa):
    """Parse the perf_analyzer output to extract throughput and p99 latency."""
    throughputs = []
    p99_latencys = []
    for i in range(num_numa):
        max_throughput = -999999999999
        max_p99_latency = -999999999999
        with open(os.path.join(meta_path, "modelzoo", "inference_log", f"{model_name}_client{i}.txt"), "r") as f:
            for line in f.readlines():
                if "throughput:" in line:
                    print(line)
                    res = line.strip().split()
                    if max_throughput < float(res[3]):
                        max_throughput = float(res[3])
                    if max_p99_latency < float(res[3]):
                        max_p99_latency = float(res[3])
        throughputs.append(max_throughput)
        p99_latencys.append(max_p99_latency)
    return throughputs, p99_latencys


def run_test(image, serving_path, meta_path, intra, inter, enable_xla, enable_onednn, test_method="entire"):
    numa_info = get_numa_info()
    print(numa_info)
    if test_method == "entire":
        client_numa = [v for k, v in numa_info.items()]
        server_numa = client_numa
        print(f"server_numa: {server_numa}")
        print(f"client_numa: {client_numa}")
    else:
        # 使用numa0做serve, numa1做client
        server_numa = [numa_info["node0"]]
        client_numa = [numa_info["node1"]]
    measurement_interval = [12000, 12000, 12000, 12000, 10000]

    model_list = ["wide_and_deep", "dlrm", "deepfm", "dffm", "dssm"]
    concurrency = ["24:52:4", "60:140:4", "20:52:4", "36:64:4", "60:120:4"]
    batch = [256, 256, 512, 64, 512]
    min_cluster_size_list = [200, 300, 300, 50, 200]
    ports = [8502 + i for i in range(len(numa_info))]

    output = {}
    for i in range(len(model_list)):
        output[model_list[i]] = {}
        saved_model_path = os.path.join(meta_path, "modelzoo", model_list[i], "result")
        for item in os.listdir(saved_model_path):
            item_path = os.path.join(saved_model_path, item)
            if os.path.isdir(item_path):
                saved_model_path = item_path
                break

        for item in os.listdir(saved_model_path):
            item_path = os.path.join(saved_model_path, item)
            if os.path.isdir(item_path) and item != "eval":
                saved_model_path = item_path
                break

        start_server_and_client(meta_path=meta_path, concurrency=concurrency[i], batch=batch[i],
                                client_numa=client_numa, server_numa=server_numa,
                                measurement_interval=measurement_interval[i], model_name=model_list[i],
                                model_path=saved_model_path, serving_path=serving_path, ports=ports, intra=intra,
                                inter=inter, min_cluster_size=min_cluster_size_list[i], enable_xla=enable_xla,
                                enable_onednn=enable_onednn, image=image)


def boolean_string(string):
    low_string = string.lower()
    if low_string not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return low_string == 'true'


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_method",
                        help="single numa or entire machine",
                        type=str,
                        default="entire")
    parser.add_argument("--serving_path",
                        help="set the path of TF-serving",
                        type=str)
    parser.add_argument("--image",
                        help="the image name of tritonclient",
                        type=str,
                        default="nvcr.io/nvidia/tritonserver:24.05-py3-sdk")
    parser.add_argument("--meta_path",
                        help="the full path of modelzoo .ie modelzoo location",
                        type=str)
    parser.add_argument("--intra",
                        help="tensorflow_intra_op_parallelism",
                        type=int,
                        default=1)
    parser.add_argument("--inter",
                        help="tensorflow_inter_op_parallelism",
                        type=int,
                        default=-1)
    parser.add_argument("--enable_XLA",
                        help="enable XLA",
                        type=boolean_string,
                        default=False)
    parser.add_argument("--enable_oneDNN",
                        help="enable oneDNN",
                        type=boolean_string,
                        default=False)

    return parser


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    if args.test_method == "entire" or args.test_method == "single":
        run_test(args.image, args.serving_path, args.meta_path, args.intra, args.inter, args.enable_XLA,
                 args.enable_oneDNN, args.test_method)
    else:
        print("测试方法参数有误，请输入'entire'或'single''")
