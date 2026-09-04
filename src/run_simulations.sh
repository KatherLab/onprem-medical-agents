#!/bin/bash
NUM_RUNS=5

for i in $(seq 1 $NUM_RUNS)
do
   echo "================================="
   echo "Launching simulation run #$i"
   echo "================================="
   # Call simulate.py and pass run_id through the command line.
   # python ./src/simulate.py --run_id $i
   python -m src.simulate --run_id $i
done

echo "All simulations completed."