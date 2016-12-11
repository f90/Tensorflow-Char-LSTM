import tensorflow as tf

def makeSequence(sequence, labels):
    '''
    Generates a SequenceExample out of a sequence of inputs and outputs
    :param sequence: Sequence input
    :param labels: Sequence output
    :return: SequenceExample
    '''
    assert(len(sequence) == len(labels)) # Assume RNN step-wise compatible input/output

    context = tf.train.Features(feature={
        "length" : _int64_feature(len(sequence))
    })

    feature_lists = tf.train.FeatureLists(feature_list={
        "inputs" : _bytes_feature_list(sequence),
        "outputs" : _bytes_feature_list(labels)
    })

    ex = tf.train.SequenceExample(
        context=context, feature_lists=feature_lists
    )

    return ex

def loadSequence(path):
    '''
    Load SequenceExample from a file in a given path
    :param path: path to file
    :return: (parsed context, parsed sequence)
    '''

    # Read in SequenceExamples
    filename_queue = tf.train.string_input_producer([path],
                                                    num_epochs=None)

    reader = tf.TFRecordReader()
    key, serialized_example = reader.read(filename_queue)

    # Define how to parse the example
    context_features = {
        "length": tf.FixedLenFeature([], dtype=tf.int64)
    }
    sequence_features = {
        "inputs": tf.FixedLenSequenceFeature([], dtype=tf.string), # String => bytearray
        "outputs": tf.FixedLenSequenceFeature([], dtype=tf.string)
    }

    # Parse the example (returns a dictionary of tensors)
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=serialized_example,
        context_features=context_features,
        sequence_features=sequence_features
    )

    # Convert string/byte to int representation
    sequence_parsed["inputs"] = tf.string_to_number(sequence_parsed["inputs"], tf.int32)
    sequence_parsed["outputs"] = tf.string_to_number(sequence_parsed["outputs"], tf.int32)
    return (key, context_parsed, sequence_parsed)

def writeSequences(seqs, path):
    '''
    Write list of SequenceExamples to file in path
    :param seqs: list of SequenceExamples
    :param path: File path
    :return:
    '''
    with path.NamedTemporaryFile() as fp:
        writer = tf.python_io.TFRecordWriter(fp.name)
        for seq in seqs:
            writer.write(seq.SerializeToString())
        writer.close()
        print("Wrote to {}".format(fp.name))

def _int64_feature(value):
  """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))


def _int64_feature_list(values):
  """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _bytes_feature_list(values):
  """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])