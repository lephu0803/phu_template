import argparse
import logging
import os

import tensorflow as tf 

from model.input_fn import input_fn
from model.utils import Params
from model.utils import set_logger
from model.utils import save_dict_to_json
from model.model_fn import model_fn
from model.training import train_and_evaluation

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiment/test',
                    help='directory contain params.json')
parser.add_argument('--data_dir', default='data/64x64_SIGNS')
parser.add_argument('--restore_from', default=None, help='Optional, directory or file containing weights to reload before training')

if __name__ == '__main__':
    tf.set_random_seed()
    args = parser.parse_args()

    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), 'No json configuration file found at {}'.format(json_path)
    params = Params(json_path)

    # Check not overwriting previous version

    model_dir_has_bent_weights = os.path.isdir(os.path.join(args.model_dir, 'best_weights'))
    overwritting = model_dir_has_bent_weights and args.restore_from is None
    assert not overwritting, "Weights found in model_dir, abouting to avoid overwrite"

    set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the data pineline
    logging.info("Creating the datasset...")
    data_dir = args.data_dir
    train_data_dir = os.path.join(data_dir, 'train_signs')
    dev_data_div = os.path.join(data_dir, "dev_signs")

    # Get the filename from the train and dev sets
    train_filenames = [os.path.join(train_data_dir, f) for f in os.listdir(train_data_dir)
                        if f.endswith('.jpg')]
    eval_filenames = [os.path.join(dev_data_div, f) for f in os.listdir(dev_data_div)
                        if f.endswith('.jpg')]
    
    train_labels = [int(f.split('/')[-1][0]) for f in train_filenames]
    eval_labels = [int(f.split('/')[-1][0]) for f in eval_filenames]

    params.train_size = len(train_filenames)
    params.eval_size = len(eval_labels)

    train_inputs = input_fn(True, train_filenames, train_labels, params)
    eval_inputs = input_fn(False, eval_filenames, eval_labels, params)

    # Defines the model
    logging.info("Creating the model...")
    train_model_spec = model_fn('train', train_inputs, params)
    eval_model_spec = model_fn('eval', eval_inputs, params)

    logging.info('Starting trainig for {} epochs'.format(params.num_epochs))
    train_and_evaluation(train_model_spec, eval_model_spec, args.model_dir, params, args.restore_from)

