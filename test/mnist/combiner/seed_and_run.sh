#!/bin/bash
#sleep 1
MODEL_ID=$(python3 init_model.py | sed -n -e 's/^.*Created seed model with id: //p')
echo "found model:" $MODEL_ID
#sleep 1
echo "starting:"
fedn run fedavg -s $MODEL_ID -r 5
echo "stopping..."
