import tensorflow as tf
import time
import os
import threading
import sacred

import Test
import Sample
import Utils
import Dataset
from Model import LyricsPredictor

ex = sacred.Experiment("lyrics_language_model")

@ex.config
def hyper_config():
    # TRAINING
    train_settings = dict()
    train_settings['trainPerc'] = 0.9 # Percentage of examples that will be used for training
    train_settings["checkpoint_dir"] = "checkpoints" # Directory for model checkpoints
    train_settings["checkpoint_path"] = os.path.join(train_settings["checkpoint_dir"], "lyrics_model")
    train_settings["log_dir"] = "log" # Directory for Tensorboard logging
    train_settings["max_iterations"] = 200 # Maximum number of training iterations
    train_settings["learning_rate_decay_epoch"] = 100 # Frequency of learning rate decay (steps)
    train_settings["learning_rate_decay_factor"] = 0.9 # Learning rate is multiplied by this every learning_rate_decay_epoch steps
    train_settings["save_model_epoch_frequency"] = 100 # Frequency of checkpoint generation (steps)
    train_settings["learning_rate"] = 0.01 # Initial learning rate

    #DATA
    data_settings = {"input_csv" : "/mnt/daten/Datasets/Million_Lyrics/metrolyrics_3artist.csv", # Input CSV
                     "input_vocab" : "/mnt/daten/Datasets/Million_Lyrics/metrolyrics.vocab"} # Input Vocabulary (created if it does not exist yet)

    #INPUT PIPELINE
    input_settings = dict()
    input_settings["num_enqueue_threads"] = 10 # Number of threads to provide RNN input
    input_settings["queue_capacity"] = 5000 # Maximum number of examples in the Input queue
    input_settings["min_queue_capacity"] = 2000 # Minimum number of examples in the Input queue to ensure enough randomness

    #MODEL
    model_settings = dict()
    model_settings["l2_regularisation"] = 0.0 # According to Karpathy, this is often not needed
    model_settings["batch_size"] = 128
    model_settings["num_unroll"] = 100 # How many steps the RNN should be unrolled
    model_settings["lstm_size"] = 512 # Hidden state size
    model_settings["num_layers"] = 1 # Number of LSTM cells
    model_settings["input_dropout"] = 0.2 # Dropout for LSTM cell inputs
    model_settings["output_dropout"] = 0.2 # Dropout for LSTM cell outputs

@ex.main
def run_experiment(train_settings, data_settings, input_settings, model_settings):
    Utils.prepareFolders(train_settings)
    Utils.cleanup(train_settings)
    train(train_settings, data_settings, input_settings, model_settings)
    Test.test(train_settings, data_settings, input_settings, model_settings)
    Sample.sample(train_settings, data_settings, input_settings, model_settings)

def train(train_settings, data_settings, input_settings, model_settings):
    # DATA
    data, vocab = Dataset.readLyrics(data_settings["input_csv"], data_settings["input_vocab"])
    trainIndices, testIndices = Dataset.createPartition(data, train_settings["trainPerc"])

    # INPUT PIPELINE
    keyInput = tf.placeholder(tf.string) # To identify each sequence
    lengthInput = tf.placeholder(tf.int32) # Length of sequence
    seqInput = tf.placeholder(tf.int32, shape=[None]) # Input sequence
    seqOutput = tf.placeholder(tf.int32, shape=[None]) # Output sequence

    q = tf.RandomShuffleQueue(input_settings["queue_capacity"], input_settings["min_queue_capacity"],
                              [tf.string, tf.int32, tf.int32, tf.int32])
    enqueue_op = q.enqueue([keyInput, lengthInput, seqInput, seqOutput])

    with tf.device("/cpu:0"):
        key, contextT, sequenceIn, sequenceOut = q.dequeue()
        context = {"length" : tf.reshape(contextT, [])}
        sequences = {"inputs" : tf.reshape(sequenceIn, [contextT]),
                    "outputs" : tf.reshape(sequenceOut, [contextT])}

    # MODEL
    model = LyricsPredictor(model_settings,
                            vocab.size + 1) # Output has to additionally support EOS token
    model.inference(key, context, sequences, input_settings["num_enqueue_threads"])
    loss = model.loss(model_settings["l2_regularisation"])[0]

    # TRAINING OPS
    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0.0))
    # Learning rate
    initial_learning_rate = tf.constant(train_settings["learning_rate"])
    learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, train_settings["learning_rate_decay_epoch"], train_settings["learning_rate_decay_factor"])
    tf.summary.scalar("learning_rate", learning_rate)

    # Gradient calculation
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars, aggregation_method=2), # Use experimental aggregation to reduce memory usage
                                      5.0)

    # Visualise gradients
    vis_grads =  [0 if i == None else i for i in grads]
    for g in vis_grads:
        tf.summary.histogram("gradients_" + str(g), g)

    optimizer = tf.train.AdamOptimizer(learning_rate)

    train_op = optimizer.apply_gradients(zip(grads, tvars),
                                         global_step=global_step)

    trainOps = [loss, train_op,
               global_step, learning_rate]

    # Start session
    sess = tf.Session()
    coord = tf.train.Coordinator()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    tf_threads = tf.train.start_queue_runners(sess=sess, coord=coord)


    # CHECKPOINTING
    #TODO save model directly after every epoch, so that we can safely refill the queues after loading a model (uniform sampling of dataset is still ensured)
    # Load pretrained model to continue training, if it exists
    latestCheckpoint = tf.train.latest_checkpoint(train_settings["checkpoint_dir"])
    if latestCheckpoint is not None:
          restorer = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2)
          restorer.restore(sess, latestCheckpoint)
          print('Pre-trained model restored')

    saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2)

    # Enqueueing method in different thread, loading sequence examples and feeding into FIFO Queue
    def load_and_enqueue(indices):
        run = True
        key = 0 # Unique key for every sample, even over multiple epochs (otherwise the queue could be filled up with two same-key examples)
        while run:
            for index in indices:
                current_seq = data[index][1]
                try:
                    sess.run(enqueue_op, feed_dict={keyInput: str(key),
                                              lengthInput: len(current_seq)-1,
                                            seqInput: current_seq[:-1],
                                            seqOutput: current_seq[1:]},
                                    options=tf.RunOptions(timeout_in_ms=60000))
                except tf.errors.DeadlineExceededError as e:
                    print("Timeout while waiting to enqueue into input queue! Stopping input queue thread!")
                    run = False
                    break
                key += 1
            print "Finished enqueueing all " + str(len(indices)) + " samples!"

    # Start a thread to enqueue data asynchronously to decouple data I/O from training
    with tf.device("/cpu:0"):
        t = threading.Thread(target=load_and_enqueue, args=[trainIndices])
        t.start()

    # LOGGING
    # Add histograms for trainable variables.
    histograms = [tf.summary.histogram(var.op.name, var) for var in tf.trainable_variables()]
    summary_op = tf.summary.merge_all()
    # Create summary writer
    summary_writer = tf.summary.FileWriter(train_settings["log_dir"], sess.graph.as_graph_def(add_shapes=True))

    current_time = time.time()
    print("Starting training. Total number of model parameters: " + str(Utils.getTotalNumParameters()))
    loops = 0
    while loops < train_settings["max_iterations"]:
        loops += 1
        [res_loss, _, res_global_step, res_learning_rate, summary] = sess.run(trainOps + [summary_op])
        new_time = time.time()
        print("Chars per second: " + str(float(model_settings["batch_size"] * model_settings["num_unroll"]) / (new_time - current_time)))
        current_time = new_time
        print("Loss: " + str(res_loss) + ", Learning rate: " + str(res_learning_rate) + ", Step: " + str(res_global_step))

        # Write summaries for this step
        summary_writer.add_summary(summary, global_step=int(res_global_step))
        if res_global_step % train_settings["save_model_epoch_frequency"] == 0:
            print("Saving model...")
            saver.save(sess, train_settings["checkpoint_path"], global_step=int(res_global_step))

    # Stop our custom input thread
    print("Stopping custom input thread")
    sess.run(q.close())  # Then close the input queue
    t.join(timeout=1)

    # Close session, clear computational graph
    sess.close()
    tf.reset_default_graph()

results = ex.run()