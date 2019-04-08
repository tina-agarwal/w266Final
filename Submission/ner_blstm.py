
import numpy as np
import random
from keras.preprocessing.sequence import pad_sequences


def readfile(filepath):
    '''
    args: file train, test or dev.txt
    returns the below format :
    [ ['-EMPTYLINE-', 'O\n'], ['RECORD', 'O\n'], ['#995723', 'O\n'], ['042186535', 'O\n'], ['SH', 'O\n'] ]
    '''
    f = open(filepath)
    sentences = []
    sentence = []
    for line in f:
        if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
            if len(sentence) > 0:
                sentences.append(sentence)
                sentence = []
            continue
        splits = line.split(' ')
        sentence.append([splits[0], splits[-1]])

    if len(sentence) > 0:
        sentences.append(sentence)
        sentence = []
    return sentences



# ['-EMPTYLINE-', ['-', 'E', 'M', 'P', 'T', 'Y', 'L', 'I', 'N', 'E', '-'], 'O\n'], ...]
def addCharInfo(Sentences):
    for i, sentence in enumerate(Sentences):
        for j, data in enumerate(sentence):
            chars = [c for c in data[0]]
            Sentences[i][j] = [data[0], chars, data[1]]
    return Sentences



# 0-pads all words
def addpadding(Sentences):
    maxlen = 52
    for sentence in Sentences:
        char = sentence[1]
        for x in char:
            maxlen = max(maxlen, len(x))
    for i, sentence in enumerate(Sentences):
        Sentences[i][1] = pad_sequences(Sentences[i][1], 52, padding='post')
    return Sentences



# returns matrix with 1 entry = list of 3 elements:
# word indices, character indices, label indices
def createDataset(sentences, word2Idx, label2Idx, char2Idx):
    unknownIdx = word2Idx['UNKNOWN_TOKEN']
    paddingIdx = word2Idx['PADDING_TOKEN']
    dataset = []

    wordCount = 0
    unknownWordCount = 0

    for sentence in sentences:
        wordIndices = []
        charIndices = []
        labelIndices = []

        for word, char, label in sentence:
            wordCount += 1
            if word in word2Idx:
                wordIdx = word2Idx[word]
            elif word.lower() in word2Idx:
                wordIdx = word2Idx[word.lower()]
            else:
                wordIdx = unknownIdx
                unknownWordCount += 1
            charIdx = []
            for x in char:
                charIdx.append(char2Idx[x])
            # Get the label and map to int
            wordIndices.append(wordIdx)
            charIndices.append(charIdx)
            labelIndices.append(label2Idx[label])

        dataset.append([wordIndices, charIndices, labelIndices])

    return dataset


def batchGenerator(data):
    l = []
    for i in data:
        l.append(len(i[0]))
    l = set(l)
    batches = []
    batch_len = []
    z = 0
    for i in l:
        for batch in data:
            if len(batch[0]) == i:
                batches.append(batch)
                z += 1
        batch_len.append(z)
    return batches, batch_len



# define casing s.t. NN can use case information to learn patterns
def getCasing(word, caseLookup):
    casing = 'other'

    numDigits = 0
    for char in word:
        if char.isdigit():
            numDigits += 1

    digitFraction = numDigits / float(len(word))

    if word.isdigit():  # Is a digit
        casing = 'numeric'
    elif digitFraction > 0.5:
        casing = 'mainly_numeric'
    elif word.islower():  # All lower case
        casing = 'allLower'
    elif word.isupper():  # All upper case
        casing = 'allUpper'
    elif word[0].isupper():  # is a title, initial char upper, then all lower
        casing = 'initialUpper'
    elif numDigits > 0:
        casing = 'contains_digit'

    return caseLookup[casing]


# return batches ordered by words in sentence
def createEqualBatches(data):
    # num_words = []
    # for i in data:
    #     num_words.append(len(i[0]))
    # num_words = set(num_words)

    n_batches = 100
    batch_size = len(data) // n_batches
    num_words = [batch_size * (i + 1) for i in range(0, n_batches)]

    batches = []
    batch_len = []
    z = 0
    start = 0
    for end in num_words:
        # print("start", start)
        for batch in data[start:end]:
            # if len(batch[0]) == i:  # if sentence has i words
            batches.append(batch)
            z += 1
        batch_len.append(z)
        start = end

    return batches, batch_len




def iterate_minibatches(dataset, batch_len):
    start = 0
    for i in batch_len:
        tokens = []
        char = []
        labels = []
        data = dataset[start:i]
        start = i
        for dt in data:
            t, ch, l = dt
            l = np.expand_dims(l, -1)
            tokens.append(t)
            char.append(ch)
            labels.append(l)
        
        yield np.asarray(labels), np.asarray(tokens), np.asarray(char)  

    # returns data with character information in format



