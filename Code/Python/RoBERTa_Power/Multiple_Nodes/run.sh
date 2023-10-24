#!/bin/bash

# Paramemters
#actions=${1:-"template"}
actions=${1:-"monitoring"}
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
        for batch_size in 8 16 32 64
        #for batch_size in 16 32 64
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
        found=$(helm list | grep roberta-monitoring)

        if [[ ${found} = "" ]]; then
            install=true
        else
            read -n 1 -p "Helm deployment 'roberta-monitoring' was found. Do you want to redeploy it? (y/n): " answer
            echo

            if [[ ${answer} = "y" ]]; then
                install=true
                helm uninstall roberta-monitoring
            else
                install=false
            fi
        fi

        if [[ ${install} == true ]]; then
            helm upgrade --install --wait --set namespace=${namespace} --set jobName="roberta-monitoring" --set setupCommands[0]="sleep infinity" --set volumes[0].claimName=${claim_name} -f monitoring.yaml roberta-monitoring /home/metalcycling/Programs/Foundation_Model_Stack/Fork/tools/scripts/appwrapper-pytorchjob/chart
        else
            echo "Deployment untouched"
        fi

    elif [[ ${action} == "watch" ]]; then
        watch -n 0.2 "oc get pods,podgroups,pytorchjobs,appwrappers"

    else
        echo "Nothing to do"

    fi
done
