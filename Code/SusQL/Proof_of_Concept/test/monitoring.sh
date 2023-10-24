#!/bin/bash

kubectl get pods,jobs,appwrappers

echo
echo "TOTAL ENERGY"

for label_group in $(kubectl get labelgroups --no-headers -o custom-columns=":metadata.name")
do
    echo ${label_group}: $(kubectl get labelgroup ${label_group} -o=jsonpath="{.status.totalEnergy}")
done
