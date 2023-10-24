"""
Cluster monitoring script
"""

# %% Modules

import os
import sys
import json
import yaml
import math
import time
import subprocess

from datetime import datetime

# %% Functions

def bash(command, stdout = False):
    """
    - Run bash commands as if they are directly typed on the shell and return response
    - Print 'stdout' if 'stdout = True'
    """
    proc = subprocess.Popen(command, shell = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
    response = []

    for encoded in proc.stdout.readlines():
        decoded = encoded.decode("utf-8") 

        if decoded[-1] == '\n':
            response.append(decoded[:-1])
        else:
            response.append(decoded)

    if stdout:
        for line in response:
            print(line)

    return response

def get_memory_from_string(string):
    """
    Parse string with memory information. Returns the value in Gi.
    """
    if string[-2:] == "Ki":
        return float(string[:-2]) / 1.0e6

    elif string[-2:] == "Mi":
        return float(string[:-2]) / 1.0e3

    elif string[-2:] == "Gi":
        return float(string[:-2]) / 1.0e0

def get_cpu_from_string(string):
    """
    Parse string with cpu information. Returns the value in millicores.
    """
    if isinstance(string, int):
        return float(string)
    else:
        if string[-1] == "m":
            return float(string[:-1])
        elif string.isnumeric():
            return float(string) * 1000.0
        else:
            print("CPU value '%s' is not understood" % (string))
            quit()

def get_gpu_from_string(string):
    """
    Parse string with gpu information. Returns the value in number of fractional GPUs.
    """
    return float(string)

def get_dashed_header(header_string, num_header_characters = 80):
    """
    Returns a centered string with dashes on both sides separated by spaces
    """
    if len(header_string) > num_header_characters:
        output_string = header_string[:num_header_characters]
    else:
        num_dashes = num_header_characters - (len(header_string) + 2)
        num_left_dashes = math.ceil(num_dashes / 2)
        num_right_dashes = num_dashes - num_left_dashes
        output_string = "%s %s %s\n" % ("-" * num_left_dashes, header_string, "-" * num_right_dashes)

    return output_string

def find_index(array, string):
    """
    Returns the array index where the word 'string' is found first
    """
    index = -1

    for idx, line in enumerate(array):
        if string in line:
            index = idx
            break

    return index

# %% Main function

def main(namespaces, type_of_resources, monitoring = True):
    """
    Main program to monitor resources
    """

    # %% Get nodes capacity

    if "nodes" in type_of_resources:
        cluster_info = yaml.safe_load("\n".join(bash("oc get nodes -o yaml")))
        nodes = {}

        for item in cluster_info["items"]:
            name = item["metadata"]["name"]
            capacity = item["status"]["capacity"]

            nodes[name] = { "avail": {}, "alloc": {} }
            nodes[name]["avail"]["cpu"] = get_cpu_from_string(capacity["cpu"]) if "cpu" in capacity else 0.0
            nodes[name]["avail"]["gpu"] = get_gpu_from_string(capacity["nvidia.com/gpu"]) if "nvidia.com/gpu" in capacity else 0.0
            nodes[name]["avail"]["mem"] = get_memory_from_string(capacity["memory"]) if "memory" in capacity else 0.0

    # %% Print cluster data

    while True:
        t_start = time.time()

        stdout = ""

        for resource_name in type_of_resources:
            if not resource_name == "nodes":
                stdout += get_dashed_header(resource_name.upper() + ":")
                responses = []

                for idx, namespace in enumerate(namespaces):
                    stdout += "NAMESPACE: '%s'.\n" % (namespace)
                    response = bash("oc -n %s get %s" % (namespace, resource_name))

                    if len(response) == 0:
                        stdout += "No resources found in namespace\n"
                    else:
                        for line in response:
                            stdout += line + "\n"

                    if idx < len(namespaces) - 1:
                        stdout += "\n"

                stdout += "\n"

            else:
                stdout += get_dashed_header("CLUSTER NODES:")

                for name in nodes.keys():
                    node_info = bash("oc describe node %s | grep \"Allocated resources:\" -A 9" % (name))

                    cpu_idx = find_index(node_info, "cpu")
                    gpu_idx = find_index(node_info, "nvidia.com/gpu")
                    mem_idx = find_index(node_info, "memory")

                    nodes[name]["alloc"]["cpu"] = get_cpu_from_string(node_info[cpu_idx].split()[1]) if cpu_idx >= 0 else 0.0
                    nodes[name]["alloc"]["gpu"] = get_gpu_from_string(node_info[gpu_idx].split()[1]) if gpu_idx >= 0 else 0.0
                    nodes[name]["alloc"]["mem"] = get_memory_from_string(node_info[mem_idx].split()[1]) if mem_idx >= 0 else 0.0

                    cpu_avail = nodes[name]["avail"]["cpu"] 
                    gpu_avail = nodes[name]["avail"]["gpu"] 
                    mem_avail = nodes[name]["avail"]["mem"] 

                    cpu_alloc = nodes[name]["alloc"]["cpu"] 
                    gpu_alloc = nodes[name]["alloc"]["gpu"] 
                    mem_alloc = nodes[name]["alloc"]["mem"] 

                    stdout += "Node: %s\n" % (name)
                    stdout += "    cpu: %8.2f / %8.2f (%6.2f%%)\n" % (cpu_alloc, cpu_avail, cpu_alloc / cpu_avail * 100.0 if cpu_avail > 0.0 else 0.0)
                    stdout += "    gpu: %8.2f / %8.2f (%6.2f%%)\n" % (gpu_alloc, gpu_avail, gpu_alloc / gpu_avail * 100.0 if gpu_avail > 0.0 else 0.0)
                    stdout += "    mem: %8.2f / %8.2f (%6.2f%%)\n" % (mem_alloc, mem_avail, mem_alloc / mem_avail * 100.0 if mem_avail > 0.0 else 0.0)
                    stdout += "\n"

        t_stop = time.time()

        stdout += "Aggregation time: %.2f\n" % (t_stop - t_start)
        stdout += "Current time: %s\n" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        if not monitoring:
            print(stdout[:-1])
            break
        else:
            os.system("clear")
            print(stdout[:-1])

# %% Main program

if __name__ == "__main__":
    assert(len(sys.argv) > 1)

    try:
        namespaces = ["default"]

        if len(sys.argv) >= 2:
            type_of_resources = sys.argv[1].split(",")

        if len(sys.argv) >= 3:
            namespaces = sys.argv[2].split(",")

        main(namespaces, type_of_resources, True)

    except KeyboardInterrupt:
        try:
            quit()
        except SystemExit:
            quit()

# %% End of program
