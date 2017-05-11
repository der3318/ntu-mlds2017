#!/bin/bash

# initialization script, prepare data
# $1: directory to faces jpg

if [[ -z $1 ]]; then
    echo "usage: ./init.sh <data_dir>";
    exit;
fi

set -e

mkdir -p data/

python3 preprocessing.py $1

wget https://github.com/ryankiros/skip-thoughts/archive/master.zip
unzip master.zip
rm master.zip

mv skip-thoughts-master skip-thoughts
mkdir -p skip-thoughts/models

bash download.sh
cd data
unzip data.zip
cd ..
