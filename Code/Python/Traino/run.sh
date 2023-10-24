#!/bin/bash
script=traino_2.py

for device in "cpu" "gpu"
do
    if [ ${device} == "cpu" ]; then
        sed -i "19s/.*/device = torch.device(\"cpu\")/g" ${script}
    else
        sed -i "19s/.*/device = torch.device(\"cuda:0\")/g" ${script}
    fi

    python3 ${script} > ${device}.dat
done
