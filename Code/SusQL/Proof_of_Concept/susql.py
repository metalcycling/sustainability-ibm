"""
SusQL: Sustainability Querying
"""

# %% Modules

import os
import yaml
import time
import threading
import subprocess

from datetime import datetime
from prometheus_api_client import PrometheusConnect, MetricsList, Metric
from prometheus_api_client.utils import parse_datetime

from kubernetes import config as k8s_config
from kubernetes import client

# %% Functions

def main():
    """
    Main function
    """

    # %%

    sampling_time = 1
    prom_url = "http://192.168.39.83:32395"
    prom_client = PrometheusConnect(url = prom_url)

    k8s_config.load_kube_config()
    k8s_client = client.CoreV1Api()

    while True:
        # Get the groupings requested from the configuration file
        config_file = "config.yaml"
        groupings = yaml.safe_load(open(config_file, "r"))["groupings"]

        # Collect existing data from groupings. If it is non existent create it.
        energy_data = []

        for gid, grouping in enumerate(groupings):
            filepath = grouping["filepath"]

            if os.path.isfile(filepath):
                energy_data.append(yaml.safe_load(open(filepath, "r")))

                for container_id in energy_data[gid]["containers"]:
                    energy_data[gid]["containers"][container_id] *= -1.0 # A negative value, indicates the container is not alive. We used this to check if containers are finished.

            else:
                labels = grouping["labels"]

                energy_data.append({})
                energy_data[gid]["labels"] = { label["key"]: label["value"] for label in labels }
                energy_data[gid]["containers"] = {}
                energy_data[gid]["accumulated_energy"] = 0.0
                energy_data[gid]["delta_energy"] = [0.0]
                energy_data[gid]["total_energy"] = [0.0]
                energy_data[gid]["timestamp"] = [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]

                yaml.safe_dump(energy_data[gid], open(filepath, "w"))

        # Collect metrics for all containers involved in the set
        t_start = time.time()

        for gid, grouping in enumerate(groupings):
            filepath = grouping["filepath"]
            labels = grouping["labels"]

            # Get all pods belonging to the label set
            label_selector = ",".join("%s=%s" % (label["key"], label["value"]) for label in labels)
            pods = [pod.metadata.name for pod in k8s_client.list_pod_for_all_namespaces(label_selector = label_selector).items]

            print("Pods in label set (%s): %s" % (label_selector, str(pods)))

            # Gather metrics for these pods
            if len(pods) > 0:
                label_selector = "|".join(pods)
                containers = prom_client.custom_query("kepler_container_joules_total{mode=\"dynamic\",pod_name=~\"%s\"}" % (label_selector))

                for container in containers:
                    metric = container["metric"]
                    value = container["value"]
                    container_id = metric["container_id"]

                    energy_data[gid]["containers"][container_id] = float(value[-1])

            # Compute delta energy and update the accumulated energy if necessary
            delta_energy = 0.0
            accumulated_energy = energy_data[gid]["accumulated_energy"]

            containers = energy_data[gid]["containers"].copy()

            for container_id, value in containers.items():
                if value < 0.0:
                    accumulated_energy += (-1.0) * value # Used '-1.0' here for clarity (as opposed to using '-=' in the update)
                    energy_data[gid]["containers"].pop(container_id)
                else:
                    delta_energy += value

            energy_data[gid]["accumulated_energy"] = accumulated_energy
            energy_data[gid]["delta_energy"].append(delta_energy)
            energy_data[gid]["total_energy"].append(accumulated_energy + delta_energy)
            energy_data[gid]["timestamp"].append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            # Save current state of the metrics
            yaml.safe_dump(energy_data[gid], open(filepath, "w"))

        t_stop = time.time()

        print("Processing time: %.2f" % (t_stop - t_start))
        print()

        time.sleep(sampling_time)

# %% Main program

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        try:
            quit()
        except SystemExit:
            quit()

# %% End of program
