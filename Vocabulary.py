import codecs
import json
import os

class Vocabulary:
    # VOCABULARY FOR CONVERSION BETWEEN CHARS AND INDICES

    def __init__(self, vocab_index_dict, index_vocab_dict, size):
        '''
        Creates a new vocabulary object, based on translation dictionaries.
        :param vocab_index_dict: Symbol to integer index mapping
        :param index_vocab_dict: Integer to symbol mapping
        :param size: Number of symbols. This can differ from the size of the dictionaries, as the indices do not have to start from zero.
        '''
        self.vocab_index_dict = vocab_index_dict
        self.index_vocab_dict = index_vocab_dict
        self.size = size

    @classmethod
    def create_from_text(cls, text, startIndex=1):
        '''
        Create a new vocabulary from all the unique characters occuring in text. Starts with the index startIndex
        :param text: Text from which to extract symbols
        :param startIndex: Index that the first symbol will be assigned to (default 1)
        :return: Vocabulary
        '''
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
        '''
        Loads vocabulary from file at path vocab_file.
        :param vocab_file: Path to vocabulary file
        :param encoding: Encoding of contents.
        :return: Vocabulary
        '''
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
        '''
        Saves this vocabulary to vocab_file.
        :param vocab_file: Path to vocabulary file.
        :param encoding: Encoding of file.
        :param overwrite: If existing files at this path should be overwritten (default: False)
        '''
        if not os.path.exists(vocab_file) or overwrite:
            with codecs.open(vocab_file, 'w', encoding=encoding) as f:
                json.dump(self.vocab_index_dict, f, indent=2, sort_keys=True)
        else:
            print("WARNING: Could not save vocabulary file, as it already exists at " + str(vocab_file))

    def index2char(self, index):
        '''
        Turn a 1-hot encoding  over the possible
        characters back into its character representation.
        :param index: Single integer or list of integers
        :return: Corresponding string
        '''
        if isinstance(index, list):
            return [self.index_vocab_dict[i] for i in index]
        else:
            return self.index_vocab_dict[index]

    def char2index(self, input):
        '''
        Convert char input to output according to dictionary
        :param input: Single character or string
        :return: Integer index or integer array
        '''
        if isinstance(input, list) or isinstance(input, str):
            return [self.vocab_index_dict[c] for c in input]
        else:
            return self.vocab_index_dict[input]