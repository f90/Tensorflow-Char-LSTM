import tensorflow as tf
import numpy as np
import random
import argparse
import time
import os
from tensorflow.python.util import nest

import SequenceHandler
import Dataset
from Model import LyricsPredictor
import matplotlib.pyplot as plt

import threading

def test(train_settings, data_settings, input_settings, model_settings):
    print("Testing model...")
    # DATA
    data, vocab = Dataset.readLyrics(data_settings["input_csv"], data_settings["input_vocab"])
    trainIndices, testIndices = Dataset.createPartition(data, train_settings["trainPerc"])

    # INPUT PIPELINE
    keyInput = tf.placeholder(tf.string)
    lengthInput = tf.placeholder(tf.int32)
    seqInput = tf.placeholder(tf.int32, shape=[None])
    seqOutput = tf.placeholder(tf.int32, shape=[None])

    q = tf.FIFOQueue(input_settings["queue_capacity"],
                              [tf.string, tf.int32, tf.int32, tf.int32])
    enqueue_op = q.enqueue([keyInput, lengthInput, seqInput, seqOutput])

    with tf.device("/cpu:0"):
        key, contextT, sequenceIn, sequenceOut = q.dequeue()
        context = dict()
        context["length"] = tf.reshape(contextT, [])
        sequences = dict()
        sequences["inputs"] = tf.reshape(sequenceIn, [contextT])
        sequences["outputs"] = tf.reshape(sequenceOut, [contextT])

    # MODEL
    model = LyricsPredictor(model_settings, vocab.size + 1) # Include EOS token
    model.inference(key, context, sequences, input_settings["num_enqueue_threads"])
    loss, mean_cross_loss, sum_cross_loss, cross_loss, output_num = model.loss(model_settings["l2_regularisation"])

    # Monitor the loss.
    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0.0))

    # LOOP
    # Start a prefetcher in the background, initialize variables
    sess = tf.Session()
    tf.train.start_queue_runners(sess=sess)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # CHECKPOINTING
    # Load pretrained model to test
    latestCheckpoint = tf.train.latest_checkpoint(train_settings["checkpoint_dir"])
    restorer = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2)
    restorer.restore(sess, latestCheckpoint)
    print('Pre-trained model restored')

    # Enqueueing method in different thread, loading sequence examples and feeding into FIFO Queue
    def load_and_enqueue(indices):
        for index in indices:
            current_seq = data[index][1]
            sess.run(enqueue_op, feed_dict={keyInput: str(index),
                                          lengthInput: len(current_seq)-1,
                                        seqInput: current_seq[:-1],
                                        seqOutput: current_seq[1:]})
        print "Finished enqueueing all " + str(len(indices)) + " samples!"
        sess.run(q.close())

    # Start a thread to enqueue data asynchronously, and hide I/O latency.
    with tf.device("/cpu:0"):
        t = threading.Thread(target=load_and_enqueue, args=[testIndices])
        t.start()

    # LOGGING
    # Add histograms for trainable variables.
    histograms = [tf.summary.histogram(var.op.name, var) for var in tf.trainable_variables()]
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter("log", sess.graph.as_graph_def(add_shapes=True))

    inferenceOps = [loss, mean_cross_loss, sum_cross_loss, cross_loss, output_num]

    current_time = time.time()
    logprob_sum = 0.0
    character_sum = 0
    iteration = 0
    while True:
        # Step through batches, perform training or inference...
        try:
            [l, mcl, scl, cl, nb] = sess.run(inferenceOps)
        except tf.errors.OutOfRangeError:
            print("Finished testing!")
            break

        new_time = time.time()
        print("Chars per second: " + str(
            float(model_settings["batch_size"] * model_settings["num_unroll"]) / (new_time - current_time)))
        current_time = new_time

        logprob_sum += scl # Add up per-char log probabilities of predictive model: Sum_i=1^N (log_2 q(x_i)), which is equal to cross-entropy term for all chars
        character_sum +=  nb # Add up how many characters were in the batch

        print(l, mcl, scl)
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, global_step=int(iteration))
        iteration += 1

        print("Bit-per-character: " + str(logprob_sum / character_sum))

    # Close session, clear computational graph
    sess.close()
    tf.reset_default_graph()
