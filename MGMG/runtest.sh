#!/bin/bash

declare -a allGoalParams=("H0" "Om0" "Xi0" "n" "R0" "lambdaRedshift" "alpha" "beta" "ml" "sl" "mh" "sh" )

baseName="testAll"

i=1 
for par in ${allGoalParams[@]};do
    echo $par
    OUT = configTest$par
    
    cat <<EOF >$OUT
    
    param='$par'
    fout=$baseName$par
    nObsUse=None
    nSamplesUse=None
    nInjUse=None
    npoints=5
 
    EOF
    
    echo $baseName$par
    
    pids[${i}]=$!
    i=$((i+1))
    python testAllparams.py --config=$OUT &
done    

for pid in ${pids[*]}; do
        wait $pid
done
    


for par in ${allGoalParams[@]};do
    echo removing $par
    rm $OUT
done