#!/bin/bash
NUM_RUNS=5
DATASET="all" # Or another dataset you want to evaluate.

for i in $(seq 1 $NUM_RUNS)
do
   echo "================================="
   echo "Evaluating run #$i for dataset ${DATASET}"
   echo "================================="
   python -m src.evaluate --dataset ${DATASET} --run_id $i
done