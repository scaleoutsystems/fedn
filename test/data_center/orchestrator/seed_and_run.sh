#!/bin/bash
sleep 5
MODEL_ID=$(python3 init_model.py | sed -n -e 's/^.*Created seed model with id: //p')
echo "found model:" $MODEL_ID
sleep 15
echo "starting:"
fedn run fedavg -s $MODEL_ID -r 10
echo "stopping..."
