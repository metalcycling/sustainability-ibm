#!/bin/bash

function measure()
{
    timestamp=$(date "+%F-%T-%Z")
    num_gpus=$(nvidia-smi -L | wc -l)
    gpus_power_data=$(nvidia-smi --query --display=POWER)

    mapfile -t gpus_power_draw <<< $(echo "${gpus_power_data}" | grep "Power Draw")
    mapfile -t gpus_power_limit <<< $(echo "${gpus_power_data}" | grep "  Power Limit")

    echo -n ${timestamp} ""

    for gpu_id in $(seq 0 $((${num_gpus} - 1)))
    do
        power_draw=$(echo ${gpus_power_draw[${gpu_id}]} | awk '{print $4}')
        power_limit=$(echo ${gpus_power_limit[${gpu_id}]} | awk '{print $4}')
        echo -n ${power_draw}/${power_limit} ""
    done

    echo
}

while true
do
    measure
    sleep ${1}
done
