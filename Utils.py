import shutil
import os
import tensorflow as tf

def getTotalNumParameters():
    '''
    Returns the total number of parameters contained in all trainable variables
    :return: Number of parameters (int)
    '''
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            #print(dim)
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters

def convert_to_dict(tuple_state, dict_state=None, i=0):
    '''
    Converts an LSTM state in a nested tuple to a dictionary
    :param tuple_state: LSTM state in the form of a nested tuple
    :param dict_state: Starting dictionary (normally empty)
    :param i: current iteration: Don't input this yourself!
    :return: Dictionary with (name, tensor) entries
    '''
    if dict_state is None:
        dict_state = dict()
    for elem in tuple_state:
        if isinstance(elem, tuple): # Recursively go down nested tuple
            dict_state, i = convert_to_dict(elem, dict_state, i)
        else:
            dict_state["lstm_state_" + str(i)] = elem
            i += 1
    return dict_state, i

def cleanup(train_settings, model=True, log=True, dataset=False):
    if model:
        shutil.rmtree(train_settings["checkpoint_dir"])
        os.makedirs(train_settings["checkpoint_dir"])
    if log:
        shutil.rmtree(train_settings["log_dir"])
        os.makedirs(train_settings["log_dir"])
    if dataset:
        if os.path.exists("datasetSplit.pkl"):
            os.remove("datasetSplit.pkl")

def prepareFolders(train_settings):
    if not os.path.exists(train_settings["checkpoint_dir"]):
        os.makedirs(train_settings["checkpoint_dir"])
    if not os.path.exists(train_settings["log_dir"]):
        os.makedirs(train_settings["log_dir"])