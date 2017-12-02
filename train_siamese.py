import tensorflow as tf
import numpy as np
from siamese import SiameseRNN
import random
import pickle
from dataset_dynamic import *
import os
import sys

np.random.seed(17)
#python train_siamese.py --embedding_dim 300 --dropout 0.1 --l2_reg_lambda 0.0 --hidden_units 30 --max_sequence_length 20 --batch_size 32 --max_steps 50000 --optimizer adam --learning_rate 0.01 --rnn_type lstm


# Parameters
# ==================================================

tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 300)")
tf.flags.DEFINE_float("dropout", 0.5, "Dropout probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")
#tf.flags.DEFINE_string("training_files", "person_match.train2", "training file (default: None)")
tf.flags.DEFINE_integer("hidden_units", 50, "Number of hidden units in softmax regression layer (default:50)")
tf.flags.DEFINE_integer("max_sequence_length", 20, "Max sequence length")
tf.flags.DEFINE_string("rnn_type", 'lstm', "Type of RNN (lstm, gru, ...)")
tf.flags.DEFINE_integer("num_rnn_layers", 1, "Num of RNN stacked layers")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("max_steps", 100000, "Number of training epochs (default: 200)")
#tf.flags.DEFINE_integer("num_epochs", 300, "Number of training epochs (default: 200)")
#tf.flags.DEFINE_integer("evaluate_every", 1000, "Evaluate model on dev set after this many steps (default: 100)")
#tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_string("optimizer", "adam", "Training algorithm to use(sgd, adam, adadelta, ...")
tf.flags.DEFINE_float("learning_rate", 0.01, "Learning rate to use in the training algorithm")
tf.flags.DEFINE_string("weight_init_mode", None, "mode of weight initialization (xavier, normal, uniform)")
tf.flags.DEFINE_float("weight_init_scale", 0.1, "scale of weight initialization (eg std for normal)")
tf.flags.DEFINE_string("loss", "1", "loss function to minimize(contrastive1, contrastive2, manhattan")

# Misc Parameters
tf.flags.DEFINE_float("distance_threshold", 0.5, "Distance Threshold to consider same question or not")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

#if FLAGS.training_files==None:
    #print"Input Files List is empty. use --training_files argument."
    #exit()
WEIGHT_INITIALIZATION_DICT = {'xavier' : lambda x: tf.contrib.layers.xavier_initializer(), # Xavier initialisation
                              'normal' : lambda x: tf.random_normal_initializer(stddev=x), # Initialization from a standard normal
                              'uniform': lambda x: tf.random_uniform_initializer(minval =-x, maxval=x) # Initialization from a uniform distribution
                             }
    
# Model parameters
model_parameters = {}
model_parameters['rnn_type'] = FLAGS.rnn_type
model_parameters['dropout'] = FLAGS.dropout
model_parameters['padding'] = 'right'
model_parameters['sequence_length'] = FLAGS.max_sequence_length
model_parameters['opt'] = FLAGS.optimizer
model_parameters['learning_rate'] = FLAGS.learning_rate
model_parameters['max_steps'] = FLAGS.max_steps
model_parameters['n_input'] = FLAGS.embedding_dim
model_parameters['n_hidden'] = FLAGS.hidden_units
model_parameters['rnn_layers'] = FLAGS.num_rnn_layers
model_parameters['batch_size'] = FLAGS.batch_size
model_parameters['distance_threshold'] = FLAGS.distance_threshold
model_parameters['loss'] = FLAGS.loss

if FLAGS.weight_init_mode is None:
    model_parameters['weight_initializer'] = None
else:
    model_parameters['weight_initializer'] = WEIGHT_INITIALIZATION_DICT[FLAGS.weight_init_mode](FLAGS.weight_init_scale)

 


#Create data set
ds = DataSet_dynamic(FLAGS.max_sequence_length, 8, 2)


#Create tensorflow model and train
print('Create model...')
model = SiameseRNN(model_parameters)
print('Train model...')
model.train(ds)
    









    
    
    