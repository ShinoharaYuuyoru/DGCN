def ReadDescriptions():
    specialSymbolSet = set()

    allDescData = dict()
    with open("WN18/definitions.txt") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            formLine = line.split('\t')     # id, entity, description

            # Process description.
            formLine[2] = formLine[2].lower()       # Change all words to lower case
            formLine[2] = formLine[2].encode('ascii', 'ignore').decode('ascii')     # Change all special symbols to ascii

            # Check how many normal symbols
            #   We find there are a lot of symbols out of English character.
            #       But we decided to reserve all these spcecial symbols, except some normal character like ',' '.' '!', etc.
            for chara in formLine[2]:
                if chara < 'a' or chara > 'z':
                    specialSymbolSet.add(chara)
            # Delete useless slash symbols.
            formLine[2] = DeleteUselessNormalSymbols(formLine[2])

            allDescData[formLine[0]] = formLine[2]
        f.close()
    
    print(len(allDescData))
    print(specialSymbolSet)

    return allDescData

def DeleteUselessNormalSymbols(description):
    # Delete
    description = description.replace(',', '')
    description = description.replace('.', '')
    description = description.replace(':', '')
    description = description.replace(';', '')
    description = description.replace('!', '')
    description = description.replace('?', '')

    # Replace by ' '
    description = description.replace('(', ' ')
    description = description.replace(')', ' ')
    description = description.replace('`', '\'')
    description = description.replace('/', ' ')

    # description = description.replace('$', ' ')
    # description = description.replace('%', ' ')
    # description = description.replace('=', ' ')
    # description = description.replace('+', ' ')

    # Reserve:
    #   ''', '"', '-', '_', ' ', '*', '^'.

    return description

def ReadWN18RRDataset():
    allWN18RREntities = set()
    allWN18RRRelations = set()

    with open("WN18RR/train.txt") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            formLine = line.split('\t')     # id, relation, id

            allWN18RREntities.add(formLine[0])
            allWN18RREntities.add(formLine[2])
            allWN18RRRelations.add(formLine[1])
        f.close()
    
    with open("WN18RR/valid.txt") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            formLine = line.split('\t')     # id, relation, id

            allWN18RREntities.add(formLine[0])
            allWN18RREntities.add(formLine[2])
            allWN18RRRelations.add(formLine[1])
        f.close()

    with open("WN18RR/test.txt") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            formLine = line.split('\t')     # id, relation, id

            allWN18RREntities.add(formLine[0])
            allWN18RREntities.add(formLine[2])
            allWN18RRRelations.add(formLine[1])
        f.close()
    
    print(len(allWN18RREntities))
    print(len(allWN18RRRelations))

    return allWN18RREntities, allWN18RRRelations

def FilterEntites(allDescData, allWN18RREntities):
    filteredDescData = dict()

    for entity in allWN18RREntities:
        if entity in allDescData:
            filteredDescData[entity] = allDescData[entity]
        else:
            print("ERROR: Entity lost in description data!")
            print('\t' + entity)
            print('\t' + "This entity's description will be filled as empty string.")
            filteredDescData[entity] = ""
    
    print(len(filteredDescData))

    return filteredDescData

def SaveFilteredDesc(filteredDescData):
    with open("WN18RR/description_rr.txt", 'w') as f:
        for entity in filteredDescData:
            descWordNum = len(filteredDescData[entity].split())
            f.write("{entity}\t{descWordNum}\t{entityDesc}\n".format(entity=entity, descWordNum=descWordNum, entityDesc=filteredDescData[entity]))
        f.close()

def CreateWN18RRDict(allWN18RREntities, allWN18RRRelations):
    entityId = 0
    with open("WN18RR/entityId.dict", 'w') as f:
        for entity in allWN18RREntities:
            # print(entity + '\t' + entityId)
            # print("{entity}\t{entityId}".format(entity=entity, entityId=entityId))
            f.write("{entity}\t{entityId}\n".format(entity=entity, entityId=entityId))

            entityId += 1
        f.close()
    
    relationId = 0
    with open("WN18RR/relationId.dict", 'w') as f:
        for relation in allWN18RRRelations:
            # print("{relation}\t{relationId}".format(relation=relation, relationId=relationId))
            f.write("{relation}\t{relationId}\n".format(relation=relation, relationId=relationId))

            relationId += 1
        f.close()

def ReadWN18RRDescription():
    allWN18RRDesc = dict()
    allWN18RRDescWordNum = dict()

    with open("WN18RR/description_rr.txt") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            formLine = line.split('\t')     # id, descWordNum, desc

            allWN18RRDesc[formLine[0]] = formLine[2]
            allWN18RRDescWordNum[formLine[0]] = int(formLine[1])
        f.close()
    
    print(len(allWN18RRDesc))

    return allWN18RRDesc, allWN18RRDescWordNum

if __name__ == "__main__":
    # definitions.txt contains all entities with descriptions, and the entities' id.
    allDescData = ReadDescriptions()

    # Read all entities and relations in WN18RR:
    #   40943 entities, 11 relations.
    # BUG: We observed that WN18RR is a version of simply deleting reversal relations.
    #   But this caused that some entites in valid.txt and test.txt are not appeared in train.txt.
    #   As now, we don't know this is a true problem or not. Observations in experiment step needed.
    allWN18RREntities, allWN18RRRelations = ReadWN18RRDataset()

    # Get filtered description data.
    filteredDescData = FilterEntites(allDescData, allWN18RREntities)

    # Save filtered description data to file.
    #   Actually, all 40943 entities have the descriptions.
    SaveFilteredDesc(filteredDescData)

    # Create WN18RR entity & relaion dictionary.
    CreateWN18RRDict(allWN18RREntities, allWN18RRRelations)

    # Read filtered description file.
    allWN18RRDesc, allWN18RRDescWordNum = ReadWN18RRDescription()
