import csv
import re
import cPickle as pickle
import os
import random

from Vocabulary import Vocabulary

import SequenceHandler

def readLyrics(csvPath, vocabPath, indices=None, convert2IndicesAndAddEOS=True):
    '''
    Reads in lyrics csv, optionally returns only a subset
    :param csvPath:
    :return:
    '''

    # Create vocabulary (mapping between characters and ints)
    vocab = Vocabulary.create_from_text('abcdefghijklmnopqrstuvwxyz0123456789 \n')

    # Save vocab
    vocab.save(vocabPath)

    wordCount = dict()
    totalWordCount = 0

    # Read in CSV lyrics line by line, build Tensor for each
    data = list()
    with open(csvPath, 'r') as csvFile: # Open CSV
        # Prepare CSV
        reader = csv.reader(csvFile, delimiter=",")
        firstRow = reader.next()
        lyricsIndex = firstRow.index("lyrics")
        artistIndex = firstRow.index("artist")

        # Go through CSV
        i=0
        for row in reader:
            i+=1
            if(i%10000==0): print i
            # Prepare lyrics
            seq = preprocessLyrics(row[lyricsIndex]) # Append and preprocess

            if convert2IndicesAndAddEOS:
                # Convert chars to int
                seq = vocab.char2index(seq)
                # Add sequence-end token:
                seq.append(vocab.size)

            # Write lyrics
            data.append([row[artistIndex].lower().strip(), seq])

    if indices is not None:
        return data[indices], vocab
    else:
        return data, vocab


def preprocessLyrics(lyrics):
    '''
    Removes all special characters and leaves only a-z, numbers 0-9 as well as space and line breaks
    :param lyrics
    :return:
    '''

    lyrics = lyrics.lower()
    expression = re.compile('[^a-z0-9 \n]+', re.UNICODE)
    return re.sub(expression, "", lyrics)

def createPartition(data, trainPerc, partitionPath='datasetSplit.pkl'):
    createNew = True
    if os.path.exists(partitionPath):
        with open(partitionPath, 'r') as file:
            perc, indices = pickle.load(file)
        createNew = (perc != trainPerc)
    if createNew:
        print("Creating new dataset partition!")
        # Split data into train and test sets. Condition: Songs of the same artist have to be all EITHER in the training OR the test set.
        uniqueArtists = set([x[0] for x in data])
        print("Computing training artists")
        trainArtists = random.sample(tuple(uniqueArtists), int(round(trainPerc * float(len(uniqueArtists)))))
        trainIndices = list()
        testIndices = list()
        print("Assigning indices")
        for i in range(0, len(data)):
            if data[i][0] in trainArtists:
                trainIndices.append(i)
            else:
                testIndices.append(i)
        indices = [trainIndices, testIndices]

        # Save indices for later usage (e.g. evaluation)
        with open(partitionPath, 'wb') as file:
            pickle.dump([trainPerc, indices], file)
    return indices