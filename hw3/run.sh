python3 preprocessing.py --test $1
mkdir -p ./samples
mkdir -p ./model
wget -P ./model/ https://www.dropbox.com/s/d2fv4j29pp3rhlp/model_after_epoch_598_dcgan.ckpt.data-00000-of-00001
wget -P ./model/ https://www.dropbox.com/s/pyt6ydnhbhvhxtn/model_after_epoch_598_dcgan.ckpt.index
wget -P ./model/ https://www.dropbox.com/s/ynf01d09equ8frx/model_after_epoch_598_dcgan.ckpt.meta
python3 test.py --testing_text $1 --resume_model ./model/model_after_epoch_598_dcgan.ckpt
