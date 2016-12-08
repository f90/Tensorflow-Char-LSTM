import codecs
import json
import numpy
import os

class Vocabulary:
    # VOCABULARY FOR CONVERSION BETWEEN CHARS AND INDICES

    def __init__(self, vocab_index_dict, index_vocab_dict, size):
        self.vocab_index_dict = vocab_index_dict
        self.index_vocab_dict = index_vocab_dict
        self.size = size

    @classmethod
    def create_from_text(cls, text, startIndex=1):
        unique_chars = list(set(text))
        size = len(unique_chars) + startIndex
        vocab_index_dict = {}
        index_vocab_dict = {}
        for i, char in enumerate(unique_chars):
            vocab_index_dict[char] = i + startIndex  # Reserve 0 for "no input/output" during padding of RNN
            index_vocab_dict[i + startIndex] = char  # Reserve 0 for "no input/output" during padding of RNN
        return cls(vocab_index_dict, index_vocab_dict, size)

    @classmethod
    def load(cls, vocab_file, encoding="UTF8"):
        with codecs.open(vocab_file, 'r', encoding=encoding) as f:
            vocab_index_dict = json.load(f)
        index_vocab_dict = {}
        max_index = 0
        for char, index in vocab_index_dict.iteritems():
            index_vocab_dict[index] = char
            max_index = max(index, max_index)
        vocab_index_dict = vocab_index_dict
        index_vocab_dict = index_vocab_dict
        size = max_index + 1
        return cls(vocab_index_dict, index_vocab_dict, size)

    def save(self, vocab_file, encoding="UTF8", overwrite=False):
        if not os.path.exists(vocab_file) or overwrite:
            with codecs.open(vocab_file, 'w', encoding=encoding) as f:
                json.dump(self.vocab_index_dict, f, indent=2, sort_keys=True)
        else:
            print("WARNING: Could not save vocabulary file, as it already exists at " + str(vocab_file))

    # HELPERS
    def batches2string(self, batches):
        """Convert a sequence of batches back into their (most likely) string representation."""
        s = [''] * batches[0].shape[0]
        for b in batches:
            s = [''.join(x) for x in zip(s, self.char2index(b))]
        return s

    def index2char(self, index):
        """Turn a 1-hot encoding or a probability distribution over the possible
        characters back into its (most likely) character representation."""
        if isinstance(index, list):
            return [self.index_vocab_dict[i] for i in index]
        else:
            return self.index_vocab_dict[index]

    def char2index(self, input):
        '''
        Convert char input to output according to dictionary
        :param input: Original representation
        :return: Converted representation
        '''
        if isinstance(input, list) or isinstance(input, str):
            return [self.vocab_index_dict[c] for c in input]
        else:
            return self.vocab_index_dict[input]