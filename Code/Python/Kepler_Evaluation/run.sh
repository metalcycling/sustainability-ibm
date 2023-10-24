#!/bin/bash

#filename="roberta.yaml"
filename="idle-app.yaml"
app_name=$(grep jobName ${filename} | awk '{ print $2 }')
chart="/home/metalcycling/Programs/Foundation_Model_Stack/Master/tools/scripts/appwrapper-pytorchjob/chart"

actions="uninstall,install"
#actions="template"

for action in $(echo ${actions} | tr ',' '\n')
do
    if [[ ${action} = "uninstall" ]]; then
        helm uninstall ${app_name}

    elif [[ ${action} = "install" ]]; then
        helm upgrade --wait --install -f ${filename} ${app_name} ${chart}

    elif [[ ${action} = "template" ]]; then
        helm template --debug -f ${filename} ${chart}

    elif [[ ${action} = "copy-to" ]]; then
        echo "hi"

    else
        echo "Nothing to do."

    fi
done

#torchrun --nnodes=${WORLD_SIZE} --node_rank=${RANK} --nproc_per_node=$(nvidia-smi -L | wc -l) --rdzv_id=101 --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" train_robertaplus_torchnative.py --simulated_gpus=$(nvidia-smi -L | wc -l) --b_size=${BATCH_SIZE} --num_steps=1000 --datapath=/workspace/data/input --logdir=/workspace/data/output --report_interval=10 --reset_stepcount --vocab=50261
