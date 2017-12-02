import tensorflow as tf
import numpy as np
import time
import os
import sys


#Define RNN namespace according to tensorflow version
if tf.__version__ == '0.12.0':
    rnn_namespace = tf.nn.rnn_cell
elif tf.__version__ == '1.0.1':
    rnn_namespace = tf.contrib.rnn
    
def _seq_length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length

def _last_relevant(output, length):
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)

    return relevant
    
def get_mean_distances(distances, Y):
    idx_positives = (Y == 1)
    idx_negatives = (Y == 0)
    distances_positives = distances[idx_positives]
    distances_negatives = distances[idx_negatives]
    mean_positives = np.mean(distances_positives)
    mean_negatives = np.mean(distances_negatives)
    return mean_positives, mean_negatives

class SiameseRNN(object):


    def _create_rnn(self, input, reuse_vars=False):
    
        with tf.variable_scope('RNN', reuse=reuse_vars):
        
            # Define a lstm cell with tensorflow
            if self.parameters['rnn_type'].lower() == 'lstm':
                rnn_cell = rnn_namespace.BasicLSTMCell(self.parameters['n_hidden'], forget_bias=1.0)
            elif self.parameters['rnn_type'].lower() == 'gru':
                rnn_cell = rnn_namespace.GRUCell(self.parameters['n_hidden'])
            elif self.parameters['rnn_type'] == 'lstm2':
                rnn_cell = rnn_namespace.LSTMCell(self.parameters['n_hidden'], initializer=self.parameters['weight_initializer'])
            elif self.parameters['rnn_type'] == 'rnn':
                rnn_cell = rnn_namespace.BasicRNNCell(self.parameters['n_hidden'])

            #Add dropout
            if self.parameters['dropout'] > 0:
                rnn_cell = rnn_namespace.DropoutWrapper(rnn_cell, output_keep_prob=self.dropout_keep_prob)
                
            if self.parameters['rnn_layers'] > 1:
                rnn_cell = rnn_namespace.MultiRNNCell([rnn_cell] * self.parameters['rnn_layers'])
            
                
            outputs, states = tf.nn.dynamic_rnn(
                rnn_cell,
                input,
                dtype=tf.float32,
                sequence_length=_seq_length(input)
            )
            
            #Obtaining the correct output state
            if self.parameters['padding'].lower() == 'right': #If padding zeros is at right, we need to get the right output, since the last is not validation
                last_relevant_output = _last_relevant(outputs, _seq_length(input))
            elif self.parameters['padding'].lower() == 'left':
                last_relevant_output = outputs[:,-1,:]
            
        return last_relevant_output
        
    def get_distance(self, channel1, channel2): #I think this is the cosine similarty
        epsilon = 0.000000000000001 # to avoid nan when dividing in case of 0
        distance = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(channel1, channel2)),1,keep_dims=True))
        distance = tf.div(distance, tf.add(tf.sqrt(tf.reduce_sum(tf.square(channel1),1,keep_dims=True)),tf.sqrt(tf.reduce_sum(tf.square(channel2),1,keep_dims=True))) + epsilon)
        distance = tf.reshape(distance, [-1], name="distance")
        
        return distance
        
    def get_loss(self, y, distance, batch_size): #TODO: Test the lost implemented in DL practical 3 (this is copied from https://github.com/dhwajraj/deep-siamese-text-similarity/blob/master/siamese_network.py)
        tmp= y *tf.square(distance)
        #tmp= tf.mul(y,tf.square(d))
        tmp2 = (1-y) *tf.square(tf.maximum((1 - distance),0))
        loss = tf.reduce_sum(tmp +tmp2)/batch_size/2
        tf.summary.scalar('loss', loss)
        return loss
        
    def get_loss2(self, channel_1, channel_2, label, margin):
        """
        Defines the contrastive loss. This loss ties the outputs of
        the branches to compute the following:

               L =  Y * d^2 + (1-Y) * max(margin - d^2, 0)

               where d is the L2 distance between the given
               input pair s.t. d = ||x_1 - x_2||_2 and Y is
               label associated with the pair of input tensors.
               Y is 1 if the inputs belong to the same class in
               CIFAR10 and is 0 otherwise.

               For more information please see:
               http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

        Args:
            channel_1: output of first channel (i.e. branch_1),
                              tensor of size [batch_size, 192]
            channel_2: output of second channel (i.e. branch_2),
                              tensor of size [batch_size, 192]
            label: Tensor of shape [batch_size]
            margin: Margin of the contrastive loss

        Returns:
            loss: scalar float Tensor
        """
        ########################
        # PUT YOUR CODE HERE  #
        ########################

        d_square = tf.reduce_sum(tf.square(channel_1 - channel_2), reduction_indices=1, keep_dims=True)
        contrastive_loss = label * d_square + (1 - label) * tf.maximum(margin - d_square, 0.)
        contrastive_loss = tf.reduce_mean(contrastive_loss)
        reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        loss = tf.add(contrastive_loss,reg_loss)
        #tf.summary.scalar('Contrastive Loss', contrastive_loss)
        #tf.summary.scalar('Regularization loss', reg_loss)
        tf.summary.scalar('loss', loss)
        ########################
        # END OF YOUR CODE    #
        ########################

        return d_square, loss
        
    def get_loss3(self, channel_1, channel_2, label):
        similarities = tf.exp(-tf.reduce_sum(tf.abs(channel_1 - channel_2), axis=1))
        loss = tf.reduce_mean(tf.square(label - similarities))
        tf.summary.scalar('loss', loss)
        return similarities, loss

    def get_accuracy(self, y, predictions):
        correct_pred = tf.equal(y, predictions)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
        return accuracy
        
    def get_optimizer(self):
        #Define optimizer
        if self.parameters['opt'].lower() == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.parameters['learning_rate']).minimize(self.loss)
        elif self.parameters['opt'].lower() == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.parameters['learning_rate']).minimize(self.loss)
        elif self.parameters['opt'].lower() == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.parameters['learning_rate']).minimize(self.loss)
        elif self.parameters['opt'].lower() == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(learning_rate=self.parameters['learning_rate']).minimize(self.loss)
        elif self.parameters['opt'].lower() == 'adagraddao':
            optimizer = tf.train.AdagradDAOptimizer(learning_rate=self.parameters['learning_rate']).minimize(self.loss)
        elif self.parameters['opt'].lower() == 'momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate=self.parameters['learning_rate']).minimize(self.loss)
        elif self.parameters['opt'].lower() == 'ftrl':
            optimizer = tf.train.FtrlOptimizer(learning_rate=self.parameters['learning_rate']).minimize(self.loss)
        elif self.parameters['opt'].lower() == 'proximalgd':
            optimizer = tf.train.ProximalGradientDescentOptimizer(learning_rate=self.parameters['learning_rate']).minimize(self.loss)
        elif self.parameters['opt'].lower() == 'proximaladagrad':
            optimizer = tf.train.ProximalAdagradOptimizer(learning_rate=self.parameters['learning_rate']).minimize(self.loss)
        elif self.parameters['opt'].lower() == 'rms':
            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.parameters['learning_rate']).minimize(self.loss)
        return optimizer
            
    def __init__(self, parameters):
    
        self.parameters = parameters
        
        with tf.variable_scope('siamese') as scope:
        
            # Placeholders for input, output and dropout
            self.input_x1 = tf.placeholder(tf.float32, [None, self.parameters['sequence_length'], self.parameters['n_input']], name="input_x1")
            self.input_x2 = tf.placeholder(tf.float32, [None, self.parameters['sequence_length'], self.parameters['n_input']], name="input_x2")
            self.input_y = tf.placeholder(tf.float32, [None], name="input_y")
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            
            
            self.rnn1 = self._create_rnn(self.input_x1, reuse_vars=False)
            self.rnn2 = self._create_rnn(self.input_x2, reuse_vars=True)
            
            if self.parameters['loss'] == "contrastive1":
                self.distance = self.get_distance(self.rnn1, self.rnn2)
                with tf.name_scope("loss"):
                    self.loss = self.get_loss(self.input_y, self.distance, self.parameters['batch_size'])
            elif self.parameters['loss'] == "contrastive2":
                self.distance, self.loss = self.get_loss2(self.rnn1, self.rnn2, self.input_y, 0.5)
            elif self.parameters['loss'] == "manhattan":
                self.distance, self.loss = self.get_loss3(self.rnn1, self.rnn2, self.input_y)
            
            
            
            #Compute probability of same question
            
            #self.predictions = tf.cast(self.distance > self.parameters['distance_threshold'], tf.float32)
            #with tf.name_scope("accuracy"):
                #self.accuracy = self.get_accuracy(self.input_y, self.predictions)
            
            self.optimizer = self.get_optimizer()
        
            self.init = tf.global_variables_initializer()
            
    
            
    def train(self, ds):
    
        dropout_keep_prob = 1 - self.parameters['dropout']
        display_step = 100
        checkpoint_freq_step = 100
        #Create savers for checkpoints
        saver_last = tf.train.Saver()
        saver_best = tf.train.Saver()
        #Create checkpoint directory
        self.parameters_str = str(self.parameters['rnn_type']) + '-' + str(self.parameters['n_hidden']) + '-' + str(self.parameters['rnn_layers']) + '-' + str(self.parameters['batch_size']) + '-' + str(self.parameters['opt']) + '-' + str(self.parameters['max_steps']) + '-' + str(self.parameters['loss'])
        self.parameters_str += '-' + time.strftime("%Y%m%d-%H%M%S")
        checkpoint_dir = './checkpoints/' + self.parameters_str
        if not tf.gfile.Exists(checkpoint_dir):
            tf.gfile.MakeDirs(checkpoint_dir)
            tf.gfile.MakeDirs(checkpoint_dir + '/best_model')
            tf.gfile.MakeDirs(checkpoint_dir + '/last_model')
        self.best_loss = 150000000
    
        # Launch the graph
        with tf.Session() as sess:
            #Create summary merge for tensorboard
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter('tensorboard/Quora/' + str(self.parameters_str) + '/train', sess.graph)
            val_writer = tf.summary.FileWriter('tensorboard/Quora/' + str(self.parameters_str) + '/val', sess.graph)

            step = 1
            sess.run(self.init)
        
            # Keep training until reach max iterations
            while step * self.parameters['batch_size'] < self.parameters['max_steps']:
                total_iterations = step * self.parameters['batch_size']
                
                #Obtain batch for this iteration
                batch_x1, batch_x2, batch_y = ds.next_batch(self.parameters['batch_size'])
                # Run optimization 
                _, c = sess.run([self.optimizer, self.loss], feed_dict={self.input_x1: batch_x1, self.input_x2:batch_x2, self.input_y: batch_y, self.dropout_keep_prob: dropout_keep_prob})

        
                if (step % display_step == 0) or ((total_iterations + self.parameters['batch_size'])  >= (self.parameters['max_steps'] - 1)):
                    print('------------------------------------------------')
                    # Calculate batch loss
                    train_minibatch_loss, train_minibatch_distances, summary = sess.run([self.loss, self.distance, merged], 
                                                                                    feed_dict={self.input_x1: batch_x1, 
                                                                                                self.input_x2:batch_x2, 
                                                                                                self.input_y: batch_y, 
                                                                                                self.dropout_keep_prob: 1})
                                                                        
                    train_mean_pos_distance, train_mean_neg_distance = get_mean_distances(train_minibatch_distances, batch_y)
                                                                                                
                    print("Iter " + str(total_iterations) + ", Minibatch train Loss = " + 
                          "{:.6f}".format(train_minibatch_loss) + ", Minibatch train positive distances = " +
                          "{:.6f}".format(train_mean_pos_distance) + ", Minibatch train negative distances = " +
                          "{:.6f}".format(train_mean_neg_distance))
                    train_writer.add_summary(summary, total_iterations)

                    # Calculate val loss
                    self.val_loss, self.val_distances, summary = sess.run([self.loss, self.distance, merged], feed_dict={self.input_x1:ds._X_val_1, self.input_x2:ds._X_val_2, self.input_y:ds._Y_val, self.dropout_keep_prob: 1})
                    val_mean_pos_distance, val_mean_neg_distance = get_mean_distances(self.val_distances, ds._Y_val)
                    val_writer.add_summary(summary, total_iterations)
                    print("Iter " + str(total_iterations) + ", Validation  Loss = " + 
                          "{:.6f}".format(self.val_loss) + ", Validation  positive distances = " + 
                          "{:.6f}".format(val_mean_pos_distance) + ", Validation negative distances = " + 
                          "{:.6f}".format(val_mean_neg_distance))

                  
                    #If best loss save the model as best model so far
                    if self.val_loss < self.best_loss:
                        self.best_loss = self.val_loss
                        checkpoint_dir_tmp = checkpoint_dir + '/best_model/'
                        checkpoint_path = os.path.join(checkpoint_dir_tmp, 'model_best.ckpt')
                        saver_best.save(sess, checkpoint_path, global_step=total_iterations)
                        self.best_model_path = 'model_best.ckpt-' + str(total_iterations)
                        #print('-->save best model: ' + str(checkpoint_path) + ' - step: ' + str(step) + ' best_model_path: ' + str(self.best_model_path))
                    print('------------------------------------------------')
                    sys.stdout.flush()
                    
                #Save check points periodically or in last iteration
                if (step % checkpoint_freq_step == 0) or ( (total_iterations + self.parameters['batch_size'])  >= (self.parameters['max_steps'] - 1)):
                    checkpoint_dir_tmp =  checkpoint_dir + '/last_model/'
                    checkpoint_path = os.path.join(checkpoint_dir_tmp, 'last_model.ckpt')
                    saver_last.save(sess, checkpoint_path, global_step=total_iterations)
                    self.last_model_path = 'last_model.ckpt-' + str(total_iterations)
                    #print('-->save checkpoint model: ' + str(checkpoint_path) + ' - step: ' + str(step) + ' last_model_path: ' + str(self.last_model_path))
                    
                step += 1
            print("Optimization Finished!")
    
    def get_best_checkpoint_path(self):
        checkpoint_dir = './checkpoints/' + self.parameters_str
        

        #CHECK: is removing the best model sometimes
        if self.best_loss < self.val_loss:
            checkpoint_dir_tmp =  checkpoint_dir + '/best_model/'
            checkpoint_path = os.path.join(checkpoint_dir_tmp, self.best_model_path)
        else:
            checkpoint_dir_tmp =  checkpoint_dir + '/last_model/'
            checkpoint_path = os.path.join(checkpoint_dir_tmp, self.last_model_path)
            
        return checkpoint_path
        
    def get_distances_and_hidden(self, X1, X2, Y, checkpoint_path = None):
        
        if checkpoint_path is None:
            checkpoint_path = self.get_best_checkpoint_path()
            
        saver = tf.train.Saver()
        with tf.Session() as sess:
            print('load model: ' + str(checkpoint_path))
            saver.restore(sess, checkpoint_path)
        
            distances, hidden1, hidden2 = sess.run([self.distance, self.rnn1, self.rnn2], feed_dict={self.input_x1: X1, self.input_x2: X2, self.input_y: Y, self.dropout_keep_prob: 1})
            
        return distances, hidden1, hidden2
        
    def check(self, x1, checkpoint_path = None):
        
        if checkpoint_path is None:
            checkpoint_path = self.get_best_checkpoint_path()
            
        saver = tf.train.Saver()
        with tf.Session() as sess:
            print('load model: ' + str(checkpoint_path))
            saver.restore(sess, checkpoint_path)
            r1, r2 = sess.run([self.rnn1, self.rnn2], feed_dict={self.input_x1: x1, self.input_x2:x1, self.dropout_keep_prob: 1})
            assert  np.array_equal(r1, r2), (
                  "Not siamese network!")
            print(r1)
            print(r2)
        
    def predict(self, X_test_1, X_test_2):
        pass
