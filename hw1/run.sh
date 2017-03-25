#!/bin/bash

# mkdir
mkdir -p model

# download model from somewhere

# output result to file
python lstm.py $1 $2 --load

