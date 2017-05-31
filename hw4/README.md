## Training data

The training data is from [Marsan-Ma chat corpus](https://github.com/Marsan-Ma/chat_corpus)

I use the open_subtitles.

## Build dictionary and preprocess training data

Sample usage:
```
python3 preprocessing.py --corpus_path ../chat_corpus/open_subtitles.txt --save_path open_subtitles.npy --vocab_size 5000 --min_len 4

python3 main.py --use_ss
```
