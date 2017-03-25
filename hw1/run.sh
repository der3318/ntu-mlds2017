#!/bin/bash

# download model from somewhere
wget http://www.csie.ntu.edu.tw/~b03902007/model.data-00000-of-00001 -P model

# output result to file
python lstm.py $1 $2 --load

