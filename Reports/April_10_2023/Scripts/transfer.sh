#!/bin/bash

##################################################################################
#
# Script to move data from a container to local directory
#
# Arguments:
#   - $1: Name of the pod to connect to
#   - $2: Directory to get the data from
#   - $3: Comma separated list of files to ignore (e.g., "checkpoints,*.mod")
#
##################################################################################

# Transfer data from POD
temp_name=.temp.tar
path=$(dirname ${2})
directory=$(basename ${2})
exclude=""

for pattern in $(echo ${3} | tr ',' '\n')
do
    exclude="${exclude} --exclude=${pattern}"
done

oc exec ${1} -- tar ${exclude} -C ${path} -cvf ${temp_name} ${directory}
oc rsync ${1}:${temp_name} .
tar -xvf ${temp_name}
rm ${temp_name}
oc exec ${1} -- rm ${temp_name}
