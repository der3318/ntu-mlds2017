#! /bin/bash

bash models/get_model.sh

python3 test.py $1 $2
