#!/bin/bash

pod_name="roberta-monitoring-master-0"
directory="/workspace/data/pedro/roberta-plus"
exclude="checkpoints,*.mod"

bash transfer.sh ${pod_name} ${directory} ${exclude}
