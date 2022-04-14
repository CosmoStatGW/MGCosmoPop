#!/bin/bash

cdir=$(pwd)

P="data"

O1O2P="$P/O1O2"
O3aP="$P/O3a"
O3bP="$P/O3b"

mkdir $P


echo "Download O1-O2 data (85.6 MB)? y/n"
read DLO1O2

if [ $DLO1O2 == "y" ]; then 
    echo "Downloading O1-O2..."
    if [ "$(ls -A $O1O2P)" ]; then 
        	echo "Found O1-O2  data"
    else
        curl -o $P/GWTC-1_sample_release.tar.gz https://dcc.ligo.org/public/0157/P1800370/005/GWTC-1_sample_release.tar.gz
        tar -xvf $P/GWTC-1_sample_release.tar.gz -C $P
        mv $P/GWTC-1_sample_release $O1O2P
        rm $P/GWTC-1_sample_release.tar.gz
    fi
fi


echo "Download O3a data (11GB)? y/n"
read DLO1O3a

if [ $DLO1O3a == "y" ]; then 
    echo "Downloading O3a..."
    if [ "$(ls -A $O3aP)" ]; then 
        	echo "Found O3a  data"
    else
        #curl -o $P/all_posterior_samples.tar https://dcc.ligo.org/public/0169/P2000223/007/all_posterior_samples.tar
        tar -xvf $P/all_posterior_samples.tar -C $P
        mv $P/all_posterior_samples $O3aP
        rm $P/all_posterior_samples.tar
    fi
fi



echo "Download GWTC2.1 data (1 GB)? y/n"
read DLO1O3a1

if [ $DLO1O3a1 == "y" ]; then 
    echo "Downloading GWTC2.1 ..."
    pip install zenodo_get
    cd $O3aP
    zenodo_get 5117703

fi
cd $cdir


echo "Download O3b data (17.1 GB)? y/n"
read DLO1O3b

if [ $DLO1O3b == "y" ]; then 
    echo "Downloading O3b. Warning: the files are very large and Zenodo can put bandwidth limit. If this script stalls, download the files by hand from https://zenodo.org/record/5546663#.YlghA9NBwlI and put them in data/O3b"
    if [ "$(ls -A $O3bP)" ]; then 
        	echo "Found O3b  data"
    else
        mkdir $O3bP; cd $O3bP
        zenodo_get 5546663
    fi
fi
cd $cdir


echo "Downloading LVC injections... " 

curl -o $O3aP/o3a_bbhpop_inj_info.hdf https://dcc.ligo.org/public/0168/P2000217/003/o3a_bbhpop_inj_info.hdf
curl -o $O1O2P/injections_O1O2an_spin.h5 https://dcc.ligo.org/public/0171/P2000434/003/injections_O1O2an_spin.h5


echo "Downloading injections generated with MGCosmoPop for analyzing GWTC3... "
cd $P
zenodo_get 6461447
cd $cdir
tar -xvf $P/injections_GWTC3.tar.gz -C $P
tar -xvf $P/injections_mock.tar.gz -C $P
tar -xvf $P/mock_BPL_5yr_GR.tar.gz -C $P
tar -xvf $P/mock_BPL_5yr_MG.tar.gz -C $P

if ! [ "$(ls -A $O1O2P)" ]; then
    mkdir $O1O2P
fi
if ! [ "$(ls -A $O3aP)" ]; then
    mkdir $O3aP
fi
if ! [ "$(ls -A $O3bP)" ]; then
    mkdir $O3bP
fi


mv $P/injections_GWTC3/*O2* $O1O2P
mv $P/injections_GWTC3/*O3a* $O3aP
mv $P/injections_GWTC3/*O3b* $O3bP
#mv $P/*mock* $P

echo "Cleaning..."

rm -r $P/injections_GWTC3
#rm -r $P/injections_mock
#rm -r $P/mock_BPL_5yr_GR
#rm -r $P/mock_BPL_5yr_MG

rm $P/injections_GWTC3.tar.gz
rm $P/injections_mock.tar.gz
rm $P/mock_BPL_5yr_GR.tar.gz
rm $P/mock_BPL_5yr_MG.tar.gz

echo "Done."
