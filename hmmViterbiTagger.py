import math

def updateTransition(transitionCounts, prevTag, currTag):
  if prevTag not in transitionCounts:
    transitionCounts[prevTag] = {}
  if currTag not in transitionCounts[prevTag]:
    transitionCounts[prevTag][currTag] = 0
  transitionCounts[prevTag][currTag] += 1

def updateEmission(emissionCounts, tag, word):
  if tag not in emissionCounts:
    emissionCounts[tag] = {}
  if word not in emissionCounts[tag]:
    emissionCounts[tag][word] = 0
  emissionCounts[tag][word] += 1

def updateTagCount(tagCounts, tag):
  if tag not in tagCounts:
    tagCounts[tag] = 0
  tagCounts[tag] += 1

def parseTagFile(filename):
    transitionCounts = {}
    emissionCounts = {}
    tagCounts = {}

    with open(filename, "r") as f:
        prevTag = "start"

        for line in f:
            line = line.strip()

            if line == "":
                updateTransition(transitionCounts, prevTag, "end")
                prevTag = "start"
                continue

            if "\t" not in line:
                continue  # Skipping incorrect lines

            word, tag = line.split("\t")

            updateTransition(transitionCounts, prevTag, tag)
            updateEmission(emissionCounts, tag, word)
            updateTagCount(tagCounts, tag)

            prevTag = tag

    return transitionCounts, emissionCounts, tagCounts

def transitionProb(transitionCount):
    transitionProb = {}
    for prevTag in transitionCount: 
      transitionProb[prevTag] = {} 
      total = sum(transitionCount[prevTag].values())

      for currTag in transitionCount[prevTag]: 
        count = transitionCount[prevTag][currTag]
        transitionProb[prevTag][currTag] = count / total
    return transitionProb

def emissionProb(emissionCount, tagCount):
  emissionProb = {}
  for tag in emissionCount: 
    emissionProb[tag] = {}
    total = tagCount[tag]
    for word in emissionCount[tag]: 
      count = emissionCount[tag][word]
      emissionProb[tag][word] = count / total 
  return emissionProb

def safeLog(x):
    return math.log(x) if x > 0 else float('-inf')

def viterbi(sentence, transitionProb, emissionProb, tags):
    table = [{}]
    backPointer = [{}]

    fword = sentence[0]
    for tag in tags:
        tProb = transitionProb.get("start", {}).get(tag, 1e-10)
        eProb = emissionProb.get(tag, {}).get(fword, 1e-10)
        table[0][tag] = safeLog(tProb) + safeLog(eProb)
        backPointer[0][tag] = "start"

    for t in range(1, len(sentence)):
        table.append({})
        backPointer.append({})
        word = sentence[t]

        for tag in tags:
            maxProb = float('-inf')
            bestPrev = None
            eProb = emissionProb.get(tag, {}).get(word, 1e-10)
            logeProb = safeLog(eProb)

            for prevTag in tags:
                tProb = transitionProb.get(prevTag, {}).get(tag, 1e-10)
                logtProb = safeLog(tProb)
                prevLogProb = table[t-1].get(prevTag, float('-inf'))

                prob = prevLogProb + logtProb + logeProb
                if prob > maxProb:
                    maxProb = prob
                    bestPrev = prevTag

            table[t][tag] = maxProb
            backPointer[t][tag] = bestPrev if bestPrev else "NN"

    # Find best last tag
    lastProb = table[-1]
    if not lastProb or all(p == float('-inf') for p in lastProb.values()):
        return ["NN"] * len(sentence)  # fallback if nothing survives

    lastTag = max(lastProb, key=lastProb.get)

    # Backtrace
    bestPath = [lastTag]
    for t in reversed(range(1, len(sentence))):
        lastTag = backPointer[t].get(lastTag, "NN")
        bestPath.insert(0, lastTag)

    return bestPath

def readSentence(filename):
    sentences = []
    current = []
    with open(filename, "r") as f:
        for line in f:
            word = line.strip()
            if word == "":
                if current:
                    sentences.append(current)
                    current = []
            else:
                current.append(word)
    if current:
        sentences.append(current)
    return sentences


## Initial Dictionary set   
WordFileName = "POS_train.pos"
transition, emission, tagCount =  parseTagFile(WordFileName)

# print("Transition Set:", transition)
# print("Emission Set:", emission)
# print("Tag Count: ", tagCount)


transitionProbability = transitionProb(transition)
emissionProbability = emissionProb(emission, tagCount)

## Probability Set 

# print("Transition Probability:", transitionProbability)
# print("Emission Probability:", emissionProbability)


## Viterbi Algorithm Best Path 
filename = "POS_dev.words"  
tagSet = list(tagCount.keys())

# Read all sentences from the file
sentences = readSentence(filename)

# Run Viterbi on each sentence and print the result
with open("result.pos", "w") as out:
  for i, sentence in enumerate(sentences):
    try:
      tags = viterbi(sentence, transitionProbability, emissionProbability, tagSet)
    except Exception as e:
      print(f"[WARNING] Skipping sentence {i} due to error: {e}")
      tags = ["NN"] * len(sentence)

    for word, tag in zip(sentence, tags):
      out.write(f"{word}\t{tag}\n")  # ✅ write to file instead of print()
    out.write("\n")  # separate sentences with blank lines
