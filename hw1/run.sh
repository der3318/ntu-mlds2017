#!/bin/bash

# download embedding
wget https://www.csie.ntu.edu.tw/~b03902007/glove.6B.50d.txt

# calculate forward similarity (flag "--load" is optional)
python3 rnn_forward.py glove.6B.50d.txt training_data/ $1 forward_model/sim.txt --load

# calculate backward similarity (flag "--load" is optional)
python3 rnn_backward.py glove.6B.50d.txt training_data/ $1 backward_model/sim.txt --load

# output result to file
python3 final_pred.py forward_model/sim.txt backward_model/sim.txt $2

