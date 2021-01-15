import re


def ReadDescription():
    slashSymbolSet = set()
    specialSymbolSet = set()

    allDescData = dict()
    with open("FB15K/description.txt") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            formLine = line.split('\t')     # entity, description

            # Process description.
            formLine[1] = formLine[1].lower()       # Change all words to lower case
            formLine[1] = formLine[1].encode('ascii', 'ignore').decode('ascii')     # Change all special symbols to ascii

            # Check how many kinds of slash-symbols in the description.
            #   We find that there are {'\\t', '\\r', '\\n', '\\"'} slash symbols.
            #       We will delete {'\\t', '\\r', '\\n'} in DeleteUselessSymbols() function.
            #       And let '\\"' to be '\"'
            slashSymbolItr = re.finditer(r"\\.", formLine[1])
            for slashSymbol in slashSymbolItr:
                slashSymbolSet.add(slashSymbol.group())
            # Delete useless slash symbols.
            formLine[1] = DeleteUselessSlashSymbols(formLine[1])
            
            # Check how many normal symbols
            #   We find there are a lot of symbols out of English character.
            #       But we decided to reserve all these spcecial symbols, except some normal character like ',' '.' '!', etc.
            for chara in formLine[1]:
                if chara < 'a' or chara > 'z':
                    specialSymbolSet.add(chara)
            # Delete useless normal symbols
            formLine[1] = DeleteUselessNormalSymbols(formLine[1])

            allDescData[formLine[0]] = formLine[1]
        f.close()
    
    print(len(allDescData))
    print(slashSymbolSet)
    print(specialSymbolSet)

    return allDescData

def DeleteUselessSlashSymbols(description):
    description = description[1:-4]     # Delete ""@en

    description = description.replace('\\t', ' ')
    description = description.replace('\\r', ' ')
    description = description.replace('\\n', ' ')
    description = description.replace('\\"', '"')

    return description

def DeleteUselessNormalSymbols(description):
    # Delete
    description = description.replace(',', '')
    description = description.replace('.', '')
    description = description.replace(':', '')
    description = description.replace(';', '')
    description = description.replace('!', '')
    description = description.replace('?', '')

    # Replace by ' '
    description = description.replace('{', ' ')
    description = description.replace('}', ' ')
    description = description.replace('[', ' ')
    description = description.replace(']', ' ')
    description = description.replace('(', ' ')
    description = description.replace(')', ' ')
    description = description.replace('`', '\'')
    description = description.replace('/', ' ')
    description = description.replace('|', ' ')

    # description = description.replace('@', ' ')
    # description = description.replace('&', ' ')
    # description = description.replace('$', ' ')
    # description = description.replace('%', ' ')
    # description = description.replace('#', ' ')
    # description = description.replace('=', ' ')
    # description = description.replace('+', ' ')
    # description = description.replace('~', ' ')
    # description = description.replace('<', ' ')
    # description = description.replace('>', ' ')

    # Reserve:
    #   ''', '"', '-', '_', ' ', '*'.

    return description

def Read237Dataset():
    all237Entites = set()
    all237Relations = set()

    with open("FB15K-237/train.txt") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            formLine = line.split('\t')
            all237Entites.add(formLine[0])
            all237Entites.add(formLine[2])

            all237Relations.add(formLine[1])
        f.close()
    
    with open("FB15K-237/valid.txt") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            formLine = line.split('\t')
            all237Entites.add(formLine[0])
            all237Entites.add(formLine[2])

            all237Relations.add(formLine[1])
        f.close()
    
    with open("FB15K-237/test.txt") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            formLine = line.split('\t')
            all237Entites.add(formLine[0])
            all237Entites.add(formLine[2])

            all237Relations.add(formLine[1])
        f.close()
    
    print(len(all237Entites))
    print(len(all237Relations))

    return all237Entites, all237Relations

def FilterEntites(allDescData, all237Entities):
    filteredDescData = dict()

    for entity in all237Entities:
        if entity in allDescData:
            filteredDescData[entity] = allDescData[entity]
        else:
            print("ERROR: Entity lost in description data!")
            print('\t' + entity)
            print('\t' + "This entity's description will be filled with an empty string.")
            filteredDescData[entity] = ""
    print(len(filteredDescData))

    return filteredDescData

def SaveFilteredDesc(filteredDescData):
    with open("FB15K-237/description_237.txt", 'w') as f:
        for entity in filteredDescData:
            descWordNum = len(filteredDescData[entity].split())
            # print("{entity}\t{descWordNum}\t{entityDesc}".format(entity=entity, descWordNum=descWordNum, entityDesc=filteredDescData[entity]))
            f.write("{entity}\t{descWordNum}\t{entityDesc}\n".format(entity=entity, descWordNum=descWordNum, entityDesc=filteredDescData[entity]))
        f.close()

def Create237Dict(all237Entities, all237Relations):
    entityId = 0
    with open("FB15K-237/entityId.dict", 'w') as f:
        for entity in all237Entities:
            # print(entity + '\t' + entityId)
            # print("{entity}\t{entityId}".format(entity=entity, entityId=entityId))
            f.write("{entity}\t{entityId}\n".format(entity=entity, entityId=entityId))

            entityId += 1
        f.close()
    
    relationId = 0
    with open("FB15K-237/relationId.dict", 'w') as f:
        for relation in all237Relations:
            # print("{relation}\t{relationId}".format(relation=relation, relationId=relationId))
            f.write("{relation}\t{relationId}\n".format(relation=relation, relationId=relationId))

            relationId += 1
        f.close()

def Read237Description():
    all237Desc = dict()
    all237DescWordNum = dict()

    with open("FB15K-237/description_237.txt") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            formLine = line.split('\t')

            if len(formLine) == 3:
                all237Desc[formLine[0]] = formLine[2]
                all237DescWordNum[formLine[0]] = int(formLine[1])
            elif len(formLine) == 2:        # If the entity has no description...
                print(formLine)
                all237Desc[formLine[0]] = ""
                all237DescWordNum[formLine[0]] = 0
            else:
                print("FATAL ERROR: Entity has more than 1 description!")
                for item in formLine:
                    print(item)
        f.close()
    
    print(len(all237Desc))

    return all237Desc, all237DescWordNum


if __name__ == "__main__":
    # Read all description data of original FB15K dataset and delete all useless symbols like ""@en, \", \n, \t, etc.
    allDescData = ReadDescription()

    # Read all entities of FB15K-237:
    #   14541 Entites, 237 Relations.
    # BUG: We observed that FB15K-237 is a version of simply deleting reversal relations.
    #   But this caused that some entites in valid.txt and test.txt are not appeared in train.txt.
    #   As now, we don't know this is a true problem or not. Observations in experiment step needed.
    all237Entities, all237Relations = Read237Dataset()

    # Filter entites, because FB15K-237 is a subset of FB15K.
    filteredDescData = FilterEntites(allDescData, all237Entities)

    # Save filtered description data to file.
    SaveFilteredDesc(filteredDescData)

    # Create FB15K-237 entity & relaion dictionary.
    Create237Dict(all237Entities, all237Relations)

    # Read filtered description file.
    #   There are 3 entities without description, so we should specially process this...
    all237Desc, all237DescWordNum = Read237Description()
