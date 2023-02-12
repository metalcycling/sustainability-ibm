#!/bin/bash

function measure()
{
    timestamp=$(date "+%F-%T-%Z")
    num_gpus=$(nvidia-smi -L | wc -l)

    echo -n ${timestamp} ""

    for gpu_id in $(seq 0 $((${num_gpus} - 1)))
    do
        power_draw=$(nvidia-smi -i ${gpu_id} -q | grep "Power Draw" | awk '{print $4}')
        power_limit=$(nvidia-smi -i ${gpu_id} -q | grep "  Power Limit" | awk '{print $4}')
        echo -n ${power_draw}/${power_limit}
    done

    echo " "
}

while true
do
    measure
    sleep ${1}
done
