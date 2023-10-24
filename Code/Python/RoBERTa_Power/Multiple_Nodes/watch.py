"""
Watching script
"""

# %% Modules

import os
import json
import signal
import subprocess

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
    Parse string with memory information. Return the value in Gi
    """
    if string[-2:] == "Ki":
        return float(string[:-2]) / 1.0e6

    elif string[-2:] == "Mi":
        return float(string[:-2]) / 1.0e3

    elif string[-2:] == "Gi":
        return float(string[:-2]) / 1.0e0

# %% Main program

if __name__ == "__main__":
    node_name = "crc-pbwlw-master-0"
    node_allocatable = json.loads(bash("oc get node %s -o=jsonpath='{.status.allocatable}'" % (node_name))[0])

    cpu_max = node_allocatable["cpu"]
    mem_max = get_memory_from_string(node_allocatable["memory"])

    while True:

        node_allocated = bash("oc describe node %s | grep \"Allocated resources\" -A 8" % (node_name))
        cpu_use = node_allocated[4].split()[1]
        mem_use = get_memory_from_string(node_allocated[5].split()[1])

        pods = bash("oc get pods")
        pytorchjobs = bash("oc get pytorchjobs")
        appwrappers = bash("oc get appwrappers")

        cpu_use = node_allocated[4].split()[1]
        mem_use = get_memory_from_string(node_allocated[5].split()[1])

        os.system("clear")

        print("------------------------------------- PODS: ----------------------------------------")

        if len(pods) == 0:
            print("No resources found in default namespace.")
        else:
            for line in pods:
                print(line)

        print()

        print("---------------------------------- PYTORCHJOBS: -----------------------------------")

        if len(pytorchjobs) == 0:
            print("No resources found in default namespace.")
        else:
            for line in pytorchjobs:
                print(line)

        print()

        print("---------------------------------- APPWRAPPERS: -----------------------------------")

        if len(appwrappers) == 0:
            print("No resources found in default namespace.")
        else:
            for line in appwrappers:
                print(line)

        print()

        print("------------------------------- CLUSTER RESOURCES: -------------------------------")
        print("CPU: %s / %s (%1.2f%%)" % (cpu_use, cpu_max, float(cpu_use[:-1]) / float(cpu_max[:-1]) * 100.0))
        print("Memory: %1.2f / %1.2f (%1.2f%%)" % (mem_use, mem_max, mem_use / mem_max * 100.0))

# %% End of program
