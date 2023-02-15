#!/bin/bash

while true
do
    echo $(date "+%F-%T.%3N-%Z") $(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits)
    sleep ${1}
done
