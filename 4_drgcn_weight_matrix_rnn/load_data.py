def load_data(DSName):
    if DSName == "FB15K-237":
        return ReadFB15K237Dataset()
    elif DSName == "WN18RR":
        return ReadWN18RRDataset()
    else:
        print("ERROR: We only support 'FB15K-237' and 'WN18RR' dataset. Please check your input parameter.")
        exit(1)

def ReadFB15K237Dataset():
    # Return value
    trainData = []
    validData = []
    testData = []
    descDict = dict()
    descWordNumDict = dict()
    entityNum = 0
    relationNum = 0

    DSPath = "Datasets/FB15K-237/"
    descFileName = "description_237.txt"

    entityDict = dict()
    relationDict = dict()
    # Read entity and relation dictionary.
    with open(DSPath+"entityId.dict") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            formLine = line.split('\t')
            if len(formLine) == 2:
                entityDict[formLine[0]] = int(formLine[1])
            else:
                print("ERROR: FB15K-237 entityId.dict error!")
                exit(1)
        f.close()
    with open(DSPath+"relationId.dict") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            formLine = line.split('\t')
            if len(formLine) == 2:
                relationDict[formLine[0]] = int(formLine[1])
            else:
                print("ERROR: FB15K-237 relationId.dict error!")
                exit(1)
        f.close()
    
    entityNum = len(entityDict)
    relationNum = len(relationDict)

    # Read dataset.
    with open(DSPath+"train.txt") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            formLine = line.split('\t')
            if len(formLine) == 3:
                trainData.append([entityDict[formLine[0]], relationDict[formLine[1]], entityDict[formLine[2]]])
            else:
                print("ERROR: FB15K-237 train.txt error!")
                exit(1)
        f.close()
    with open(DSPath+"valid.txt") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            formLine = line.split('\t')
            if len(formLine) == 3:
                validData.append([entityDict[formLine[0]], relationDict[formLine[1]], entityDict[formLine[2]]])
            else:
                print("ERROR: FB15K-237 valid.txt error!")
                exit(1)
        f.close()
    with open(DSPath+"test.txt") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            formLine = line.split('\t')
            if len(formLine) == 3:
                testData.append([entityDict[formLine[0]], relationDict[formLine[1]], entityDict[formLine[2]]])
            else:
                print("ERROR: FB15K-237 test.txt error!")
                exit(1)
        f.close()
    with open(DSPath+descFileName) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            formLine = line.split('\t')
            if len(formLine) == 3:
                descDict[entityDict[formLine[0]]] = formLine[2]
                descWordNumDict[entityDict[formLine[0]]] = int(formLine[1])
            elif len(formLine) == 2:
                print("This entity has no description. We will set it with an empty description, which wordNum == 0.")
                print("\t"+str(formLine[0]))
                descDict[entityDict[formLine[0]]] = ""
                descWordNumDict[entityDict[formLine[0]]] = 0
            else:
                print("ERROR: FB15K-237 description_237.txt error!")
                exit(1)
    
    return trainData, validData, testData, descDict, descWordNumDict, entityNum, relationNum

def ReadWN18RRDataset():
    # Return value
    trainData = []
    validData = []
    testData = []
    descDict = dict()
    descWordNumDict = dict()
    entityNum = 0
    relationNum = 0

    DSPath = "Datasets/WN18RR/"
    descFileName = "description_rr.txt"

    entityDict = dict()
    relationDict = dict()
    # Read entity and relation dictionary.
    with open(DSPath+"entityId.dict") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            formLine = line.split('\t')
            if len(formLine) == 2:
                entityDict[formLine[0]] = int(formLine[1])
            else:
                print("ERROR: WN18RR entityId.dict error!")
                exit(1)
        f.close()
    with open(DSPath+"relationId.dict") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            formLine = line.split('\t')
            if len(formLine) == 2:
                relationDict[formLine[0]] = int(formLine[1])
            else:
                print("ERROR: WN18RR relationId.dict error!")
                exit(1)
        f.close()
    
    entityNum = len(entityDict)
    relationNum = len(relationDict)

    # Read dataset.
    with open(DSPath+"train.txt") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            formLine = line.split('\t')
            if len(formLine) == 3:
                trainData.append([entityDict[formLine[0]], relationDict[formLine[1]], entityDict[formLine[2]]])
            else:
                print("ERROR: WN18RR train.txt error!")
                exit(1)
        f.close()
    with open(DSPath+"valid.txt") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            formLine = line.split('\t')
            if len(formLine) == 3:
                validData.append([entityDict[formLine[0]], relationDict[formLine[1]], entityDict[formLine[2]]])
            else:
                print("ERROR: WN18RR valid.txt error!")
                exit(1)
        f.close()
    with open(DSPath+"test.txt") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            formLine = line.split('\t')
            if len(formLine) == 3:
                testData.append([entityDict[formLine[0]], relationDict[formLine[1]], entityDict[formLine[2]]])
            else:
                print("ERROR: WN18RR test.txt error!")
                exit(1)
        f.close()
    with open(DSPath+descFileName) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            formLine = line.split('\t')
            if len(formLine) == 3:
                descDict[entityDict[formLine[0]]] = formLine[2]
                descWordNumDict[entityDict[formLine[0]]] = int(formLine[1])
            else:
                print("ERROR: WN18RR description_rr.txt error!")
                exit(1)
    
    return trainData, validData, testData, descDict, descWordNumDict, entityNum, relationNum
