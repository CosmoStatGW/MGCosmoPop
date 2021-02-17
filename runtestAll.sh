#!/bin/bash

declare -a allGoalParams=("Om0" "R0" )  #("H0" "Om0" "Xi0" "n" "R0" "lambdaRedshift" "alpha" "beta" "ml" "sl" "mh" "sh" )

i=1 
for par in ${allGoalParams[@]};do
    echo $par
    pids[${i}]=$!
    i=$((i+1))
    python testAllparams.py --param=$par &
done    

for pid in ${pids[*]}; do
        wait $pid
done
    


for par in ${allGoalParams[@]};do
    echo removing $par
    rm models$par.py 
    rm getLambda$par.py
done