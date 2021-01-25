import load_data
import numpy as np
import matplotlib.pyplot as plt

def createWordDict(descDict):
    wordDict = dict()
    wordId = 0      # 0 for 0 padding in embedding.

    for entity in descDict:
        desc = descDict[entity]
        desc = desc.split()
        for word in desc:
            if word not in wordDict:
                wordDict[word] = wordId
                wordId += 1
    
    return wordDict, len(wordDict)

def processDescDict(descDict, wordDict):
    newDescDict = {}
    for entity in descDict:
        desc = descDict[entity]
        desc = desc.split()
        descWordId = np.zeros(len(desc), dtype=np.int32)
        for idx, word in enumerate(desc):
            descWordId[idx] = wordDict[word] + 1        # 0 for padding
        newDescDict[entity] = descWordId
    
    return newDescDict

trainData, validData, testData, descDict, descWordNumDict, entityNum, relationNum = load_data.load_data('WN18RR')

num_nodes = entityNum
num_rels = relationNum
wordDict, wordNum = createWordDict(descDict)
descDict = processDescDict(descDict, wordDict)


maxDescWordNum = max(descWordNumDict.values())
sta = np.zeros(maxDescWordNum + 1, dtype=np.int32)
for val in descWordNumDict.values():
    sta[val] += 1       # MAX: FB15K-237 704, WN18RR 91

plt.plot(sta)
plt.savefig("WN18RRDis.png")
s = 0
for idx, a in enumerate(sta):
    if s > (sum(sta)*0.99):
        print(idx)      # FB15K-237: 331, WN18RR:40
        break
    else:
        s += a

length = [len(descDict[i]) for i in descDict]

length = np.sort(length)
maxL = np.max(length)
minL = np.min(length)
avg = np.mean(length)
ninenine = length[int(0.99*len(length))]

a = 0
