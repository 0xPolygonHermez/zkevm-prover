#!/bin/bash

rm -rf ./zkprover.log

for i in $(seq 1 1)
do
       date
       echo "zkProver run $i started -----------------" >> ./zkprover.log
       nohup ./build-gpu/zkProver -c testvectors/config_runFile_BatchProof.json > result-$i.txt 2>&1 &
       sleep 180
       pid=$(ps -e | grep zkProver | head -n 1 | awk '{printf $1}')
       stp=""
       while [ -z "$stp" ]; do
	       kill -SIGINT $pid
	       sleep 5
	       # pid=$(ps -e | grep zkProver | head -n 1 | awk '{printf $1}')
	       stp=$(cat result-$i.txt | grep "PROVER_BATCH_PROOF done")
       done
       kill -9 $pid
       sleep 10
       echo "--------------------end--------------------" >> ./zkprover.log
done
