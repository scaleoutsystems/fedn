#!/bin/bash
#sleep 1
MODEL_ID=$(python3 init_model.py | sed -n -e 's/^.*Created seed model with id: //p')
echo "found model:" $MODEL_ID
sleep 5
echo "starting:"
#fedn run fedavg -s $MODEL_ID -r 5
fedn run fedavg -d discovery -p 8080 -t e9a3cb4c5eaff546eec33ff68a7fbe232b68a192
echo "stopping..."
