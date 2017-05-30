import sys
import re
import numpy as np
np.random.seed(3318)
from style import cprint, pprint
from data_util import DataUtil
from model import S2SModel
import tensorflow as tf
tf.set_random_seed(3318)

# config
dataDir = "data"
indexFile = "data/words.npy"
maxLen = 20
nWords = 8000
batchSize = 128
nEpochs = 100
lr = 1e-3

# get data utility
cprint("I/O", "Indexing", "red")
du = DataUtil()
if "--train" in sys.argv:
    du.buildIndex(dataDir)
    du.saveIndex(indexFile)
du.loadIndex(indexFile)
print( "len(du.words) |", len(du.words) )
print("du.words[:10] |", du.words[:10])
print("du.word2Idx[\"happy\"] |", du.word2Idx["happy"])

# load data
if "--train" in sys.argv:
    cprint("I/O", "Loading Data", "red")
    trainQ, trainA = du.loadData(dataDir, _maxLen = maxLen, _nWords = nWords)
    print("trainQ.shape trainA.shape |", trainQ.shape, trainA.shape)
    print("trainQ[0, :] |", trainQ[0, :])
    print("trainA[0, :] |", trainA[0, :])

# train
cprint("Tensorflow", "Training", "red")
s2s = S2SModel(_maxLen = maxLen, _nWords = nWords)
loss, _ = s2s.buildModel(_batchSize = batchSize)
generatedWords = s2s.buildGenerator(_batchSize = 1)
optimizer = tf.train.AdamOptimizer(learning_rate = lr).minimize(loss)
config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth = True
saver = tf.train.Saver( tf.all_variables() )
sess = tf.Session(config = config)
if "--train" in sys.argv:
    sess.run( tf.global_variables_initializer() )
    for epoch in range(nEpochs):
        rndPerm = np.random.permutation(trainQ.shape[0])
        trainQ = trainQ[rndPerm, :]
        trainA = trainA[rndPerm, :]
        batchList = []
        for i in range(0, trainQ.shape[0] - batchSize, batchSize):
            _, curLoss = sess.run([optimizer, loss],
                feed_dict = {s2s.question: trainQ[i:i + batchSize, :], s2s.answer: trainA[i:i + batchSize, :]})
        pprint( epoch / nEpochs, "- epoch=" + str(epoch + 1) + "/" + str(nEpochs) + " - loss=" + str(curLoss) )
    print("")
    cprint("Tensorflow", "Saving Model", "red")
    saver.save(sess, "tf/model")
else:
    cprint("Tensorflow", "Loading Model", "red")
    latest_checkpoint = tf.train.latest_checkpoint("./tf/")
    saver = tf.train.import_meta_graph("tf/model.meta")
    saver.restore(sess, latest_checkpoint)

# test
cprint("Tensorflow", "Testing", "red")
sentence = input("Say Something (Enter \"quit\" to Exit): ")
while(sentence != "quit"):
    testQ = np.zeros( (1, maxLen) )
    testQ[0, 0] = du.word2Idx["<start>"]
    for idx, word in enumerate( re.sub("[^a-zA-Z ,]+", " ", sentence).lower().split() ):
        if (idx + 2) >= maxLen: break
        if word not in du.word2Idx: testQ[0, idx + 1] = nWords - 1
        else:   testQ[0, idx + 1] = du.word2Idx[word]
        testQ[0, idx + 2] = du.word2Idx["<end>"]
    curWords = np.array( sess.run(generatedWords, feed_dict = {s2s.question: testQ}) )
    for wordIdx in curWords[:, 0]:
        if wordIdx == 0:    continue
        if wordIdx == du.word2Idx["<end>"]:    break
        print(du.words[wordIdx - 1] + " ", end = "")
    print("")
    sentence = input("Say Something (Enter \"quit\" to Exit): ")
sess.close()

