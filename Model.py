import tensorflow as tf
import numpy as np

class LyricsPredictor:

    def __init__(self, model_settings, vocab_size, training=True):

        # Set parameters
        self.batch_size = model_settings["batch_size"]
        self.lstm_size = model_settings["lstm_size"]
        self.num_unroll = model_settings["num_unroll"]
        self.num_layers = model_settings["num_layers"]
        self.input_dropout = model_settings["input_dropout"]
        self.output_dropout = model_settings["output_dropout"]

        self.vocab_size = vocab_size

        if not training:
            self.input_dropout = 0.0
            self.output_dropout = 0.0

    def inference(self, key, context, sequences, num_enqueue_threads):
        # RNN cells and states
        cells = list()
        initial_states = dict()
        for i in range(0, self.num_layers):
            cell = tf.contrib.rnn.LSTMBlockCell(num_units=self.lstm_size) # Block LSTM version gives better performance #TODO Add linear projection option
            cell = tf.nn.rnn_cell.DropoutWrapper(cell,input_keep_prob=1-self.input_dropout, output_keep_prob=1-self.output_dropout)
            cells.append(cell)
            initial_states["lstm_state_c_" + str(i)] = tf.zeros(cell.state_size[0], dtype=tf.float32)
            initial_states["lstm_state_h_" + str(i)] = tf.zeros(cell.state_size[1], dtype=tf.float32)
        cell = tf.nn.rnn_cell.MultiRNNCell(cells)

        # BATCH INPUT
        self.batch = tf.contrib.training.batch_sequences_with_states(
            input_key=key,
            input_sequences=sequences,
            input_context=context,
            input_length=tf.cast(context["length"], tf.int32),
            initial_states=initial_states,
            num_unroll=self.num_unroll,
            batch_size=self.batch_size,
            num_threads=num_enqueue_threads,
            capacity=self.batch_size * num_enqueue_threads * 2)
        inputs = self.batch.sequences["inputs"]
        targets = self.batch.sequences["outputs"]

        # Convert input into one-hot representation (from single integers indicating character)
        print(self.vocab_size)
        embedding = tf.constant(np.eye(self.vocab_size), dtype=tf.float32)
        inputs = tf.nn.embedding_lookup(embedding, inputs)

        # Reshape inputs (and targets respectively) into list of length T (unrolling length), with each element being a Tensor of shape (batch_size, input_dimensionality)
        inputs_by_time = tf.split(1, self.num_unroll, inputs)
        inputs_by_time = [tf.squeeze(elem, squeeze_dims=1) for elem in inputs_by_time]
        targets_by_time = tf.split(1, self.num_unroll, targets)
        targets_by_time = [tf.squeeze(elem, squeeze_dims=1) for elem in targets_by_time] # num_unroll-list of (batch_size) tensors
        self.targets_by_time_packed = tf.pack(targets_by_time) # (num_unroll, batch_size)

        # Build RNN
        state_name = initial_states.keys()
        self.seq_lengths = self.batch.context["length"]
        (self.outputs, state) = tf.nn.state_saving_rnn(cell, inputs_by_time, state_saver=self.batch,
                                                  sequence_length=self.seq_lengths, state_name=state_name, scope='SSRNN')

        # Create softmax parameters, weights and bias, and apply to RNN outputs at each timestep
        with tf.variable_scope('softmax'):
            softmax_w = tf.get_variable("softmax_w", [self.lstm_size, self.vocab_size])
            softmax_b = tf.get_variable("softmax_b", [self.vocab_size])
            logits = [tf.matmul(outputStep, softmax_w) + softmax_b for outputStep in self.outputs]

            self.logit = tf.pack(logits)

            self.probs = tf.nn.softmax(self.logit)
        tf.summary.histogram("probabilities", self.probs)
        return (self.logit, self.probs)

    def sample(self, input, current_state):
        # RNN cells and states
        cells = list()
        for i in range(0, self.num_layers):
            cell = tf.contrib.rnn.LSTMBlockCell(num_units=self.lstm_size) # Block LSTM version gives better performance #TODO Add linear projection option
            cell = tf.nn.rnn_cell.DropoutWrapper(cell,1.0,1.0) # No dropout during sampling
            cells.append(cell)
        cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        self.initial_states = cell.zero_state(batch_size=1,dtype=tf.float32)

        # Convert input into one-hot representation (from single integers indicating character)
        embedding = tf.constant(np.eye(self.vocab_size), dtype=tf.float32)
        input = tf.nn.embedding_lookup(embedding, input) # 1 x Vocab-size
        inputs_by_time = [input] # List of 1 x Vocab-size tensors (with just one tensor in it, because we just use sequence length 1

        self.outputs, state = tf.nn.rnn(cell, inputs_by_time, initial_state=current_state, scope='SSRNN')

        # Create softmax parameters, weights and bias, and apply to RNN outputs at each timestep
        with tf.variable_scope('softmax'):
            softmax_w = tf.get_variable("softmax_w", [self.lstm_size, self.vocab_size])
            softmax_b = tf.get_variable("softmax_b", [self.vocab_size])
            logits = [tf.matmul(outputStep, softmax_w) + softmax_b for outputStep in self.outputs]

            self.logit = tf.pack(logits)
            self.probs = tf.nn.softmax(self.logit)

        return self.probs, state


    def loss(self, l2_regularisation):
        with tf.name_scope('loss'):
            # Compute mean cross entropy loss for each output.
            self.cross_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logit, self.targets_by_time_packed) # (num_unroll, batchsize)

            # Mask losses of outputs for positions t which are outside the length of the respective sequence, so they are not used for backprop
            # Take signum => if target is non-zero (valid char), set mask to 1 (valid output), otherwise 0 (invalid output, no gradient/loss calculation)
            mask = tf.sign(tf.abs(tf.cast(self.targets_by_time_packed, dtype=tf.float32))) # Unroll*Batch \in {0,1}
            self.cross_loss = self.cross_loss * mask

            output_num = tf.reduce_sum(mask)
            sum_cross_loss = tf.reduce_sum(self.cross_loss)
            mean_cross_loss = sum_cross_loss / output_num # Mean loss is sum over masked losses for each output, divided by total number of valid outputs

            # L2
            vars = tf.trainable_variables()
            l2_loss = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(l2_regularisation), weights_list=vars)

            loss = mean_cross_loss + l2_loss
            tf.summary.scalar('mean_batch_cross_entropy_loss', mean_cross_loss)
            tf.summary.scalar('mean_batch_loss', loss)
        return loss, mean_cross_loss, sum_cross_loss, output_num