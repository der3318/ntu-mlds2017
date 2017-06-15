## Training data

The training data is from [Marsan-Ma chat corpus](https://github.com/Marsan-Ma/chat_corpus)

I use the open_subtitles and movie_subtitles_en.

## See result

```
bash run.sh [S2S, RL, BEST] [INPUT_FILE] [OUTPUT_FILE]
```

## Build dictionary and preprocess training data

Sample usage:
```
python3 preprocessing.py \
    --corpus_path ../chat_corpus/open_subtitles.txt \
    --corpus_path ../chat_corpus/movie_subtitles_en.txt \
    --save_path data \
    --dict_path dict.txt \
    --min_len 5 \
    --valid_ratio 0.01 \
    --vocab_size 20000
```
