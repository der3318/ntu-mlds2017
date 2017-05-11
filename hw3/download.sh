#!/bin/bash

path="skip-thoughts/models/"
<<comment1
wget -P $path http://www.cs.toronto.edu/~rkiros/models/dictionary.txt
wget -P $path http://www.cs.toronto.edu/~rkiros/models/utable.npy
wget -P $path http://www.cs.toronto.edu/~rkiros/models/btable.npy
wget -P $path http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz
wget -P $path http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz.pkl
wget -P $path http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz
wget -P $path http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz.pkl
comment1

path="data/"

wget -P $path https://www.dropbox.com/s/qztwqgz4g6e2sfs/data.zip

