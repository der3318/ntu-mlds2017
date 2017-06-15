


# Get data
wget http://homepage.ntu.edu.tw/~b03902105/MLDS/data/dict10000.txt
wget http://homepage.ntu.edu.tw/~b03902105/MLDS/data/data10000_train.npy
wget http://homepage.ntu.edu.tw/~b03902105/MLDS/data/data10000_valid.npy


if [ "$1" == "RL" ]; then
    echo "RL"
    mkdir -p RL
    wget http://homepage.ntu.edu.tw/~b03902105/MLDS/RL/model.ckpt.index
    wget http://homepage.ntu.edu.tw/~b03902105/MLDS/RL/model.ckpt.data-00000-of-00001
    wget http://homepage.ntu.edu.tw/~b03902105/MLDS/RL/model.ckpt.meta
    mv model.ckpt* RL
    python3 test.py --dict_path dict10000.txt --data_path data10000_train.npy --valid_path data10000_valid.npy --layers 3 --hidden 256 --model_path RL/model.ckpt --input_path $2 --output_path $3
fi

if [ "$1" == "S2S" ]; then
    echo "S2S"
    mkdir -p S2S
    wget http://homepage.ntu.edu.tw/~b03902105/MLDS/s2s/model.ckpt.index
    wget http://homepage.ntu.edu.tw/~b03902105/MLDS/s2s/model.ckpt.data-00000-of-00001
    wget http://homepage.ntu.edu.tw/~b03902105/MLDS/s2s/model.ckpt.meta
    mv model.ckpt* S2S
    python3 test.py --dict_path dict10000.txt --data_path data10000_train.npy --valid_path data10000_valid.npy --layers 4 --hidden 256 --model_path S2S/model.ckpt --input_path $2 --output_path $3
fi
if [ "$1" == "RL" ]; then
    echo "RL"
    mkdir -p RL
    wget http://homepage.ntu.edu.tw/~b03902105/MLDS/RL/model.ckpt.index
    wget http://homepage.ntu.edu.tw/~b03902105/MLDS/RL/model.ckpt.data-00000-of-00001
    wget http://homepage.ntu.edu.tw/~b03902105/MLDS/RL/model.ckpt.meta
    mv model.ckpt* RL
    python3 test.py --dict_path dict10000.txt --data_path data10000_train.npy --valid_path data10000_valid.npy --layers 3 --hidden 256 --model_path RL/model.ckpt --input_path $2 --output_path $3
fi

if [ "$1" == "BEST" ]; then
    echo "BEST"
    mkdir -p S2S
    wget http://homepage.ntu.edu.tw/~b03902105/MLDS/s2s/model.ckpt.index
    wget http://homepage.ntu.edu.tw/~b03902105/MLDS/s2s/model.ckpt.data-00000-of-00001
    wget http://homepage.ntu.edu.tw/~b03902105/MLDS/s2s/model.ckpt.meta
    mv model.ckpt* S2S
    python3 test.py --dict_path dict10000.txt --data_path data10000_train.npy --valid_path data10000_valid.npy --layers 4 --hidden 256 --model_path S2S/model.ckpt --input_path $2 --output_path $3
fi
