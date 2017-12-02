import tensorflow as tf
import numpy as np
import pickle
from siamese import SiameseRNN 
import pandas as pd

max_length = 15

# Create tensorflow model
checkpoint_path1 = 'checkpoints/lstm2-50-1-128-adam-7000000-manhattan-20170413-164002/best_model/model_best.ckpt-2112000'
checkpoint_path2 = 'checkpoints/lstm2-50-1-128-adam-4000000-contrastive1-20170414-111408/best_model/model_best.ckpt-3148800'


model_parameters1 = {}
model_parameters1['rnn_type'] = 'lstm2'
model_parameters1['dropout'] = 0
model_parameters1['padding'] = 'right'
model_parameters1['sequence_length'] = 15
model_parameters1['opt'] = 'adam'
model_parameters1['learning_rate'] = 0.01
model_parameters1['max_steps'] = 80000
model_parameters1['n_input'] = 300
model_parameters1['n_hidden'] = 50
model_parameters1['rnn_layers'] = 1
model_parameters1['batch_size'] = 128
model_parameters1['distance_threshold'] = 0.5
model_parameters1['loss'] = 'manhattan'
model_parameters1['weight_initializer'] = None

model_parameters2 = {}
model_parameters2['rnn_type'] = 'lstm2'
model_parameters2['dropout'] = 0
model_parameters2['padding'] = 'right'
model_parameters2['sequence_length'] = 15
model_parameters2['opt'] = 'adam'
model_parameters2['learning_rate'] = 0.01
model_parameters2['max_steps'] = 80000
model_parameters2['n_input'] = 300
model_parameters2['n_hidden'] = 50
model_parameters2['rnn_layers'] = 1
model_parameters2['batch_size'] = 128
model_parameters2['distance_threshold'] = 0.5
model_parameters2['loss'] = 'contrastive1'
model_parameters2['weight_initializer'] = None

print('Create model...')
model1 = SiameseRNN(model_parameters1)

df_test = pd.read_csv('test.csv')
test_ids = df_test['test_id'].values





start = 0
end = 0
idx = 0
for i in range(start, end):
    x_test_similarities = pd.DataFrame()

    print(i)
    # Load test pickles
    with open('pickles/X_test_1_GoogleEmbeddings_' + str(max_length) + '_file_' + str(i) + '.pickle', 'rb') as handle:
        X_test_1 = pickle.load(handle)
    with open('pickles/X_test_2_GoogleEmbeddings_' + str(max_length) + '_file_' + str(i) + '.pickle', 'rb') as handle:
        X_test_2 = pickle.load(handle) 
        
    # Compute features with similarity
    similarities_batch, h1, h2 = model1.get_distances_and_hidden(X_test_1, X_test_2, [], checkpoint_path1)
    difference_hidden_1 = np.abs(h1 - h2)
    multwise_hidden_1 = h1 * h2
        
    x_test_similarities['similarity'] = similarities_batch    
    for j in range(h1.shape[1]):
        x_test_similarities['h1_' + str(j)] = h1[:, j] 
        x_test_similarities['h2_' + str(j)] = h2[:, j]
        x_test_similarities['difference_hidden_1_' + str(j)] = difference_hidden_1[:,j]
        x_test_similarities['multwise_hidden_1_' + str(j)] = multwise_hidden_1[:,j]
     
    x_test_similarities['test_id'] = test_ids[idx:(idx + len(X_test_1))]
    
    # Save csv
    for c in x_test_similarities.columns:
        if c != 'id':
            x_test_similarities[c] = x_test_similarities[c].astype(np.float32)
    x_test_similarities.to_csv('csv/x_test_hidden_similarities_features_' + str(i) + '.csv')
        
    idx += len(X_test_1)
    
    
    
    print('Created features for file ' + str(i))

    

    
    
    
    
    
# For distances - model2
tf.reset_default_graph()
model2 = SiameseRNN(model_parameters2)

start = 0
end = 51
idx = 0
    
for i in range(start, end):
    x_test_distances= pd.DataFrame()

    print(i)
    # Load test pickles
    with open('pickles/X_test_1_GoogleEmbeddings_' + str(max_length) + '_file_' + str(i) + '.pickle', 'rb') as handle:
        X_test_1 = pickle.load(handle)
    with open('pickles/X_test_2_GoogleEmbeddings_' + str(max_length) + '_file_' + str(i) + '.pickle', 'rb') as handle:
        X_test_2 = pickle.load(handle)
        
    distances_batch, h1_2, h2_2 = model2.get_distances_and_hidden(X_test_1, X_test_2, [], checkpoint_path2)
    difference_hidden_2 = np.abs(h1_2 - h2_2)
    multwise_hidden_2 = h1_2 * h2_2
    
    x_test_distances['distances'] = distances_batch 
    
    for j in range(h1_2.shape[1]):
        x_test_distances['difference_hidden_2_' + str(j)] = difference_hidden_2[:,j]
        x_test_distances['multwise_hidden_2_' + str(j)] = multwise_hidden_2[:,j]
            
     
    x_test_distances['test_id'] = test_ids[idx:(idx + len(X_test_1))]
    
    # Save csv
    for c in x_test_distances.columns:
        if c != 'id':
            x_test_distances[c] = x_test_distances[c].astype(np.float32)
    x_test_distances.to_csv('csv/x_test_distances_features_' + str(i) + '.csv')
        
    idx += len(X_test_1)
   

    
    print('Created features for file ' + str(i))