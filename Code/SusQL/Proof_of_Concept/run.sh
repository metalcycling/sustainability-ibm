#!/bin/bash

#./susql organization-name=org-1
#./susql organization-name=org-1,group-name=grp-1
#./susql organization-name=org-1,group-name=grp-1,project-name=model-1

PROMETHEUS_URL="http://192.168.39.83:32395"

promql --host ${PROMETHEUS_URL} --start 1h 'kepler_container_core_joules_total{container_namespace="org-1", mode="dynamic"}'
