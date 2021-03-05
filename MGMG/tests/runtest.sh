#!/bin/bash

declare -a allGoalParams=("H0" "Om" )

#"Om" "Xi0" "n" "R0" "lambdaRedshift" "alpha" "beta" "ml" "sl" "mh" "sh" 

baseName="testFolder5"
basedir="../../results/$baseName"
echo $basedir
mkdir $basedir

i=1 
for par in ${allGoalParams[@]};do
    echo $par
    OUTbase=configTest$par
    OUT=$OUTbase.py
    
    cat <<EOF >$OUT
param='$par'
fout='$baseName$par'
nObsUse=10
nSamplesUse=100
nInjUse=100
npoints=5
EOF
    
    #echo $baseName$par
    #mkdir ../results/$baseName$par
    
    pids[${i}]=$!
    i=$((i+1))
    python testAllparams1.py --config=$OUTbase &
done    


for pid in ${pids[*]}; do
    wait $pid
done

for par in ${allGoalParams[@]};do
   OUTbase=configTest$par
   OUT=$OUTbase.py
   rm $OUT
   echo removed $OUT
done
 
for par in ${allGoalParams[@]};do
   mv ../../results/$baseName$par $basedir
done