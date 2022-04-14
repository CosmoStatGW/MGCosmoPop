#!/bin/bash


P="data"

O1O2P="$P/O1O2"
O3aP="$P/O3a"
O3bP="$P/O3b"

mkdir $P
#mkdir -p $O1O2P
##mkdir -p $O3aP
#mkdir -p $O3bP


echo "Download O1-O2 data (85.6 MB)? y/n"
read DLO1O2

if [ $DLO1O2 == "y" ]; then 
    echo "Downloading O1-O2..." 
    mkdir $O1O2P
    curl -o $O1O2P https://dcc.ligo.org/public/0157/P1800370/005/GWTC-1_sample_release.tar.gz
    tar -xvf $O1O2P/GWTC-1_sample_release.tar.gz
fi   