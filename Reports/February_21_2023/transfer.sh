#!/bin/bash

# Transfer data from POD
temp_name=my_data.tar
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
