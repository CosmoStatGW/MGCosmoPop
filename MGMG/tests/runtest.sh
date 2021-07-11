#!/bin/bash

declare -a allGoalParams=("H0" "Om" "Xi0" "R0" "alphaRedshift" "betaRedshift" "zp" "alpha1" "alpha2" "beta" "deltam" "ml"  "mh" "b" )

# "H0" "Om" "Xi0" "n"
# "R0" "lambdaRedshift"
# "alpha" "beta" "ml" "sl" "mh" "sh" 
#  "alpha1" "alpha2" "beta" "deltam" "ml"  "mh" "b" 


baseName="testAll_Xi0-1.8/"
basedir="../../results/$baseName"
echo $basedir
mkdir $basedir

i=1 
for par in ${allGoalParams[@]};do
    echo $par
    OUTbase=configTest$par
    OUT=$OUTbase.py
    
    cat <<EOF >$OUT
data= ['mock_BPL_5yr_aLIGOdesignSensitivity', ] #['O1O2', 'O3a']
dist_unit = 'Gpc'
param='$par'
fout='$baseName$par'
nObsUse=None
nSamplesUse=None
nInjUse=None
npoints=20
massf='broken_pow_law'
events_use = {'use': None,
          'not_use': ['GW170817', 'GW190814', 'GW190425', 'GW190426', 'GW190719', 'GW190909', 'GW190426_152155', 'GW190719_215514', 'GW190909_114149'] 
          }
spindist='skip'
rate='astro-ph'
Tobs=5.
EOF
    
    #echo $baseName$par
    #mkdir ../results/$baseName$par
    
    pids[${i}]=$!
    i=$((i+1))
    python testAllparams.py --config=$OUTbase &
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