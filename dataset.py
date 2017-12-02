from scipy import sparse
import numpy as np

class DataSet():
    """
    Utility class to handle dataset structure.
    """

    def __init__(self, X_train_1, X_train_2, Y_train, X_val_1, X_val_2, Y_val):

        assert len(X_train_1) == len(X_train_2), (
              "images.shape: {0}, labels.shape: {1}".format(str(len(X_train_1)), str(len(X_train_2))))
        assert len(X_train_1) == len(Y_train), (
              "images.shape: {0}, labels.shape: {1}".format(str(len(X_train)), str(len(Y_train))))
        assert len(X_val_1) == len(Y_val), (
              "images.shape: {0}, labels.shape: {1}".format(str(len(X_val)), str(len(Y_val))))
        assert len(X_val_1) == len(X_val_2), (
              "images.shape: {0}, labels.shape: {1}".format(str(len(X_val_1)), str(len(X_val_2))))
              
        self._num_examples = len(X_train_1)
        self._X_train_1 = X_train_1
        self._X_train_2 = X_train_2
        self._Y_train = Y_train
        self._X_val_1 = X_val_1
        self._X_val_2 = X_val_2
        self._Y_val = Y_val
        
        self._epochs_completed = 0
        self._index_in_epoch = 0


    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            print('epochs completed: ' + str(self._epochs_completed))
            self._epochs_completed += 1

            #perm = np.arange(self._num_examples)
            #np.random.shuffle(perm)
            #self._X_train_1 = self._X_train_1[perm]
            #self._X_train_2 = self._X_train_2[perm]
            #self._Y_train = self._Y_train[perm]

            start = 0
            self._index_in_epoch = batch_size
            print(self._index_in_epoch)
            assert batch_size <= self._num_examples

        end = self._index_in_epoch
        x1 = self._X_train_1[start:end]
        x2 = self._X_train_2[start:end]
        y = self._Y_train[start:end]
        return x1, x2, y
        #seq_length = self._X_train[0].toarray().shape[0]
        #n_features = self._X_train[0].toarray().shape[1]
        #n_output = self._Y_train[0].toarray().shape[1]
        #batch_x = []
        #batch_y = []
        #for x,y in zip(X,Y):
            #batch_x.append(x.toarray())
            #batch_y.append(y.toarray().reshape(n_output))
        #return np.array(batch_x), np.array(batch_y)
        
    def get_validation(self):
        return self._X_val, self._Y_val