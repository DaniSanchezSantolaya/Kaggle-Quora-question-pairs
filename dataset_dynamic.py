
import numpy as np
import pickle

data_folder = '/data/pickles/'
data_folder = 'pickles/'

class DataSet_dynamic():
    """
    Utility class to handle dataset structure.
    """

    def __init__(self, max_length, num_total_files, num_validation_file):

        assert num_validation_file != 0, ("Validation set file cannot be the first one")
        assert num_validation_file != num_total_files, ("Validation set file cannot be the last one")
              
        self.max_length = max_length
        self.num_total_files = num_total_files
        self.num_validation_file = num_validation_file

        self._actual_file = 0
        self._epochs_completed = 0
        self._index_in_file = 0
        
        
        #Load first training file
        with open(data_folder + 'X_train_1_GoogleEmbeddings_' + str(self.max_length) + '_file_' + str(self._actual_file) + '.pickle', 'rb') as handle:
            self._X_train_1 = pickle.load(handle)
        with open(data_folder + 'X_train_2_GoogleEmbeddings_' + str(self.max_length) + '_file_' + str(self._actual_file) + '.pickle', 'rb') as handle:
            self._X_train_2 = pickle.load(handle) 
        with open(data_folder + 'Y_train_GoogleEmbeddings_' + str(max_length) + '_file_' + str(self._actual_file) + '.pickle', 'rb') as handle:
            self._Y_train = pickle.load(handle) 

        self._num_samples_file = len(self._X_train_1)
            
        #Load validaiton file
        with open(data_folder + 'X_train_1_GoogleEmbeddings_' + str(self.max_length) + '_file_' + str(self.num_validation_file) + '.pickle', 'rb') as handle:
            self._X_val_1 = pickle.load(handle)
        with open(data_folder + 'X_train_2_GoogleEmbeddings_' + str(self.max_length) + '_file_' + str(self.num_validation_file) + '.pickle', 'rb') as handle:
            self._X_val_2 = pickle.load(handle) 
        with open(data_folder + 'Y_train_GoogleEmbeddings_' + str(max_length) + '_file_' + str(self.num_validation_file) + '.pickle', 'rb') as handle:
            self._Y_val = pickle.load(handle) 


    def next_batch(self, batch_size):
        start = self._index_in_file
        self._index_in_file += batch_size
        if self._index_in_file > self._num_samples_file:
            
            print('Training file ' + str(self._actual_file) + ' completed')

            # Change file
            self._actual_file += 1
            
            # If is the validation set, skip it
            if self._actual_file == self.num_validation_file:
                self._actual_file += 1
            # If we have gone through all files, start a new epoch from the first file
            if self._actual_file >= self.num_total_files:
                self._epochs_completed += 1
                print('Epoch completed! epochs completed: ' + str(self._epochs_completed))
                self._actual_file = 0
            
            print('Start with training file: ' + str(self._actual_file))
            
            #Load next training file
            with open(data_folder + 'X_train_1_GoogleEmbeddings_' + str(self.max_length) + '_file_' + str(self._actual_file) + '.pickle', 'rb') as handle:
                self._X_train_1 = pickle.load(handle)
            with open(data_folder + 'X_train_2_GoogleEmbeddings_' + str(self.max_length) + '_file_' + str(self._actual_file) + '.pickle', 'rb') as handle:
                self._X_train_2 = pickle.load(handle) 
            with open(data_folder + 'Y_train_GoogleEmbeddings_' + str(self.max_length) + '_file_' + str(self._actual_file) + '.pickle', 'rb') as handle:
                self._Y_train = pickle.load(handle) 

            start = 0
            self._index_in_file = batch_size
            #print(self._index_in_file)
            assert batch_size <= self._num_samples_file

        end = self._index_in_file
        x1 = self._X_train_1[start:end]
        x2 = self._X_train_2[start:end]
        y = self._Y_train[start:end]
        return x1, x2, y

        
    def get_validation(self):
        return self._X_val, self._Y_val