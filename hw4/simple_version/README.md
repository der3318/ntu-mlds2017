## DL HW4 Simple Version
#### Config
* data: 
	1. Marsan-Ma chat-corpus movie_subtitles_en.txt.gz top 100000 lines
	2. chatterbot corpus eng
* maxLen = 20
* nWords = 8000
* epochs = 100


#### Framework
* 2-Layered LSTM
* No Schedule-Sampling, Beam-Search or Attention-Mask


#### Train and Test
* Train - `$ python3 main.py --train`
* Test with Input Sentences - `$ python3 main.py`


#### Train and Test with RL and Attention-Mask
* Train - `$ python3 main_rl.py --train`
* Test with Input Sentences - `$ python3 main_rl.py`


