# Tensorflow-Char-LSTM
Character-based recurrent neural network for text modeling implemented in Tensorflow

This code makes use of Tensorflow Queues for proper dataset loading and processing, as well as optimised minibatch training on variable-length sequences while keeping a persistent LSTM hidden state for each sequence (stateful).
Here it is used for generating lyrics, but it can be quickly adapted to other text and even any sequence-based data.

## Requirements

Tensorflow 0.12 or later
Sacred 0.6.10 or later

## Setup

In Training.py, set the desired configuration settings. In particular, set input_csv to the CSV file with columns "lyrics" and "artist" that is used as the dataset (If you want to adapt it to your setting, change Dataset.py). Also set input_vocab to the path where you want the vocabulary file to be saved that maps character symbols to integers.

## Running an experiment

The code uses Sacred and frames one run as one experiment, in which we train a model, then test its performance on the test partition, and finally generate a sample from the model.,
This happens in run_experiment in Training.py. Be careful: Utils.cleanup removes model checkpoints and logs by default to allow another experiment when executing the code again.

In particular, Training.train performs the model training and continually saves checkpoints.
Test.test goes through the test partition once and evaluates the performance measured in bits per character.
Finally, Sample.sample generates lyrics either from scratch, or continues a text (given by current_seq in the code).

## Example output for lyrics

Coming up soon!
