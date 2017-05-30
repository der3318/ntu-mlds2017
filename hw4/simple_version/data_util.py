import os
import re
import json
import numpy as np

class DataUtil:

    def __init__(self):
        self.word2Idx = {}
        self.words = []

    def buildIndex(self, _dataDir):
        self.word2Idx = {"<start>": 1, "<end>": 2}
        self.words = ["<start>", "<end>"]
        counts = [1e10, 1e9]
        for dataFile in os.listdir(_dataDir):
            if re.match(".*\.corpus\.json", dataFile):
                with open(os.path.join(_dataDir, dataFile), "r") as fin:
                    fileData = list( json.load(fin).values() )[0]
                for part in fileData:
                    for sentence in part:
                        for word in sentence.lower().split():
                            if word in self.words:
                                counts[self.word2Idx[word] - 1] += 1
                                continue
                            self.words.append(word)
                            self.word2Idx[word] = len(self.words)
                            counts.append(1)
            if re.match(".*\.txt", dataFile):
                with open(os.path.join(_dataDir, dataFile), "r", errors = "replace") as fin:
                    for lineNum, sentence in enumerate(fin):
                        if lineNum == 100000:  break
                        for word in re.sub("[^a-zA-Z ,]+", " ", sentence).lower().split():
                            if word in self.words:
                                counts[self.word2Idx[word] - 1] += 1
                                continue
                            self.words.append(word)
                            self.word2Idx[word] = len(self.words)
                            counts.append(1)
        self.words = np.array(self.words)[ np.array(counts).argsort()[::-1] ]
        for idx, word in enumerate(self.words): self.word2Idx[word] = (idx + 1)

    def saveIndex(self, _npyFile):
        np.save(_npyFile, self.words)

    def loadIndex(self, _npyFile):
        self.words = np.load(_npyFile)
        self.word2Idx = {word: (idx + 1) for idx, word in enumerate(self.words)}

    def loadData(self, _dataDir, _maxLen = 20, _nWords = 4000):
        (dataQ, dataA) = ([], [])
        for dataFile in os.listdir(_dataDir):
            if re.match(".*\.corpus\.json", dataFile):
                with open(os.path.join(_dataDir, dataFile), "r") as fin:
                    fileData = list( json.load(fin).values() )[0]
                for part in fileData:
                    if len(part) % 2 != 0:   continue
                    for idx, sentence in enumerate(part):
                        sentenceData = [1]
                        for word in sentence.lower().split():
                            sentenceData.append(self.word2Idx[word])
                        if idx % 2 == 0:    dataQ.append(sentenceData)
                        else:   dataA.append(sentenceData)
            if re.match(".*\.txt", dataFile):
                with open(os.path.join(_dataDir, dataFile), "r", errors = "replace") as fin:
                    for lineNum, sentence in enumerate(fin):
                        if lineNum == 100000:  break
                        sentenceData = [1]
                        for word in re.sub("[^a-zA-Z ,]+", " ", sentence).lower().split():
                            sentenceData.append(self.word2Idx[word])
                        sentenceData.append(2)
                        if lineNum % 2 == 0:    dataQ.append(sentenceData)
                        else:   dataA.append(sentenceData)
        for dataIdx in range( len(dataQ) ):
            if len(dataQ[dataIdx]) >= _maxLen:
                dataQ[dataIdx] = dataQ[dataIdx][:_maxLen]
                dataQ[dataIdx][_maxLen - 1] = 2
            else:   dataQ[dataIdx] = dataQ[dataIdx] + [0 for i in range( _maxLen - len(dataQ[dataIdx]) )]
            if len(dataA[dataIdx]) >= _maxLen:
                dataA[dataIdx] = dataA[dataIdx][:_maxLen]
                dataA[dataIdx][_maxLen - 1] = 2
            else:   dataA[dataIdx] = dataA[dataIdx] + [0 for i in range( _maxLen - len(dataA[dataIdx]) )]
        dataQ = np.array(dataQ)
        dataA = np.array(dataA)
        dataQ = np.where(dataQ > (_nWords - 1), (_nWords - 1), dataQ)
        dataA = np.where(dataA > (_nWords - 1), (_nWords - 1), dataA)
        return dataQ, dataA

