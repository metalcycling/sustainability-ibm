#!/bin/bash

pod_name="monitoring-master-0"
directory="/workspace/data/pedro/output"
exclude="checkpoints,*.mod"

bash transfer.sh ${pod_name} ${directory} ${exclude}
