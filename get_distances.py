import tensorflow as tf
import numpy as np
import pickle
from siamese import SiameseRNN 

checkpoint_path = 'checkpoints/lstm-100-1-64-adam-28000000-20170407-222610/best_model/model_best.ckpt-114560'

model_parameters = {}
model_parameters['rnn_type'] = 'lstm'
model_parameters['dropout'] = 0
model_parameters['padding'] = 'right'
model_parameters['sequence_length'] = 15
model_parameters['opt'] = 'adam'
model_parameters['learning_rate'] = 0.01
model_parameters['max_steps'] = 80000
model_parameters['n_input'] = 300
model_parameters['n_hidden'] = 100
model_parameters['rnn_layers'] = 1
model_parameters['batch_size'] = 64
model_parameters['distance_threshold'] = 0.5
model_parameters['loss'] = 'manhattan'

#Create tensorflow model and train
print('Create model...')
model = SiameseRNN(model_parameters)


# Load data pickles
with open('pickles/X_1_GoogleEmbeddings_' + str(model_parameters['sequence_length']) + '.pickle', 'rb') as handle:
    X_1 = pickle.load(handle)   
with open('pickles/X_2_GoogleEmbeddings_' + str(model_parameters['sequence_length']) + '.pickle', 'rb') as handle:
    X_2 = pickle.load(handle)   
with open('pickles/Y_GoogleEmbeddings_' + str(model_parameters['sequence_length']) + '.pickle', 'rb') as handle:
    Y = pickle.load(handle)

#Check siamese model    
model.check(X_1[0].reshape([1, model_parameters['sequence_length'], model_parameters['n_input']]), checkpoint_path)

#Distances
num_samples = 1000
distances, h1, h2 = model.get_distances_and_hidden(X_1[0:num_samples], X_2[0:num_samples], Y, checkpoint_path)

print(distances.shape)
idx_positives = (Y[0:num_samples] == 1)
idx_negatives = (Y[0:num_samples] == 0)
print(idx_positives.shape)
distances_positives = distances[idx_positives]
distances_negatives = distances[idx_negatives]
mean_positives = np.mean(distances_positives)
mean_negatives = np.mean(distances_negatives)
print('Mean distance in the ' + str(len(distances_positives)) + ' positive sentences: ' + str(mean_positives))
print('Mean distance in the ' + str(len(distances_negatives)) + ' negative sentences: ' + str(mean_negatives))
