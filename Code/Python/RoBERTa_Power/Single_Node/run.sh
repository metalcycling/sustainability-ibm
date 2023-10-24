#!/bin/bash

# Paramemters
#actions=${1:-"clean,production"}
actions=${1:-"template"}
num_steps=400000
namespace=$(oc status | awk 'NR==1{print $3}')

if [[ ${namespace} = "mcad-testing" ]]; then
    claim_name=mcad-testing-pvc

elif [[ ${namespace} = "mcad-testing-prod" ]]; then
    claim_name=mcad-testing-prod-bucket

else
    echo "Running on ${namespace} is not supported"
    exit
fi

# Run actions
for action in $(echo ${actions} | tr ',' '\n')
do
    if [[ ${action} == "development" ]]; then
        helm uninstall roberta-development
        helm upgrade --install --wait --set namespace=${namespace} --set jobName="roberta-development" --set setupCommands[0]="sleep infinity" --set volumes[0].claimName=${claim_name} -f roberta.yaml roberta-development /home/metalcycling/Programs/Foundation_Model_Stack/Fork/tools/scripts/appwrapper-pytorchjob/chart

    elif [[ ${action} == "clean" ]]; then
        for name in $(helm list | grep roberta | awk '{print $1}')
        do
            helm uninstall ${name}
        done

    elif [[ ${action} == "production" ]]; then
        #for batch_size in 8 16 32 64
        for batch_size in 16 32 64
        #for batch_size in 8
        do
            sed -i "3s/.*/jobName: roberta-${batch_size}/" roberta.yaml
            sed -i "44s/.*/      value: ${batch_size}/" roberta.yaml
            sed -i "46s/.*/      value: ${num_steps}/" roberta.yaml
            helm upgrade --install --wait --set namespace=${namespace} --set volumes[0].claimName=${claim_name} -f roberta.yaml roberta-${batch_size} /home/metalcycling/Programs/Foundation_Model_Stack/Fork/tools/scripts/appwrapper-pytorchjob/chart
        done

    elif [[ ${action} == "template" ]]; then
        helm template --set namespace=${namespace} --set volumes[0].claimName=${claim_name} -f roberta.yaml /home/metalcycling/Programs/Foundation_Model_Stack/Fork/tools/scripts/appwrapper-pytorchjob/chart
    
    elif [[ ${action} == "monitoring" ]]; then
        helm uninstall roberta-monitoring
        helm upgrade --install --wait --set namespace=${namespace} --set jobName="roberta-monitoring" --set setupCommands[0]="sleep infinity" --set volumes[0].claimName=${claim_name} -f monitoring.yaml roberta-monitoring /home/metalcycling/Programs/Foundation_Model_Stack/Fork/tools/scripts/appwrapper-pytorchjob/chart

    else
        echo "Nothing to do"
    fi
done
