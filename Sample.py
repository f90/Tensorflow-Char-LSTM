import tensorflow as tf
import numpy as np

from Model import LyricsPredictor
from Vocabulary import Vocabulary


def sample(train_settings, data_settings, input_settings, model_settings):
    print("Sampling from model...")
    # Load vocab
    vocab = Vocabulary.load(data_settings["input_vocab"])

    # INPUT PIPELINE
    input = tf.placeholder(tf.int32, shape=[None], name="input") # Integers representing characters
    # Create state placeholders - 2 for each lstm cell.
    state_placeholders = list()
    initial_states = list()
    for i in range(0,model_settings["num_layers"]):
        state_placeholders.append(tuple([tf.placeholder(tf.float32, shape=[1, model_settings["lstm_size"]], name="lstm_state_c_" + str(i)), # Batch size x State size
                                    tf.placeholder(tf.float32, shape=[1, model_settings["lstm_size"]], name="lstm_state_h_" + str(i))])) # Batch size x State size
        initial_states.append(tuple([np.zeros(shape=[1, model_settings["lstm_size"]], dtype=np.float32),
                              np.zeros(shape=[1, model_settings["lstm_size"]], dtype=np.float32)]))
    state_placeholders = tuple(state_placeholders)
    initial_states = tuple(initial_states)

    # MODEL
    inference_settings = model_settings
    inference_settings["batch_size"] = 1 # Only sample from one example simultaneously
    inference_settings["num_unroll"] = 1 # Only sample one character at a time
    model = LyricsPredictor(inference_settings, vocab.size + 1)  # Include EOS token
    probs, state = model.sample(input, state_placeholders)

    # LOOP
    # Start a prefetcher in the background, initialize variables
    sess = tf.Session()
    tf.train.start_queue_runners(sess=sess)
    init_op = tf.initialize_all_variables()
    sess.run(init_op)

    # CHECKPOINTING
    # Load pretrained model to  sample
    latestCheckpoint = tf.train.latest_checkpoint(train_settings["checkpoint_dir"])
    restorer = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2)
    restorer.restore(sess, latestCheckpoint)
    print('Pre-trained model restored')

    inference = [probs, state]

    current_seq = "never"
    current_seq_ind = vocab.char2index(current_seq)

    # Warm up RNN with initial sequence
    s = initial_states
    for ind in current_seq_ind:
        # Create feed dict for states
        feed = dict()
        for i in range(0, model_settings["num_layers"]):
            for c in range(0, len(s[i])):
                feed[state_placeholders[i][c]] = s[i][c]
                feed[state_placeholders[i][c]] = s[i][c]

        feed[input] = [ind] # Add new input symbol to feed
        [p, s] = sess.run(inference, feed_dict=feed)

    # Sample until we receive an end-of-lyrics token
    iteration = 0
    while iteration < 100000:
        # Now p contains probability of upcoming char, as estimated by model, and s the last RNN state
        ind_sample = np.random.choice(range(0,vocab.size+1), p=np.squeeze(p))
        if ind_sample == vocab.size: # EOS token
            print("Model decided to stop generating!")
            break

        current_seq_ind.append(ind_sample)

        # Create feed dict for states
        feed = dict()
        for i in range(0, model_settings["num_layers"]):
            for c in range(0, len(s[i])):
                feed[state_placeholders[i][c]] = s[i][c]
                feed[state_placeholders[i][c]] = s[i][c]

        feed[input] = [ind_sample]  # Add new input symbol to feed
        [p, s] = sess.run(inference, feed_dict=feed)

        iteration += 1

    c_sample = vocab.index2char(current_seq_ind)
    print("".join(c_sample))

    sess.close()
