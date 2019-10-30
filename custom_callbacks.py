from keras.callbacks import Callback
# from data_generators import DataGeneratorDinucShuffle
from keras import backend as K
import numpy as np
from utils import progress
import sys

class SGDRScheduler(Callback):
    '''Cosine annealing learning rate scheduler with periodic restarts.

    # Usage
        ```python
            schedule = SGDRScheduler(min_lr=1e-5,
                                     max_lr=1e-2,
                                     steps_per_epoch=np.ceil(epoch_size/batch_size),
                                     lr_decay=0.9,
                                     cycle_length=5,
                                     mult_factor=1.5)
            model.fit(X_train, Y_train, epochs=100, callbacks=[schedule])
        ```

    # Arguments
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`. 
        lr_decay: Reduce the max_lr after the completion of each cycle.
                  Ex. To reduce the max_lr by 20% after each cycle, set this value to 0.8.
        cycle_length: Initial number of epochs in a cycle.
        mult_factor: Scale epochs_to_restart after each full cycle completion.

    # References
        Blog post: jeremyjordan.me/nn-learning-rate
        Original paper: http://arxiv.org/abs/1608.03983
    '''
    def __init__(self,
                 min_lr,
                 max_lr,
                 steps_per_epoch,
                 lr_decay=1,
                 cycle_length=10,
                 mult_factor=2,
                 shape="cosine"):

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_decay = lr_decay

        self.batch_since_restart = 0
        self.next_restart = cycle_length

        self.steps_per_epoch = steps_per_epoch

        self.cycle_length = cycle_length
        self.mult_factor = mult_factor
        
        self.shape = shape
        self.history = {}
        self.learning_rates = []

    def clr(self):
        '''Calculate the learning rate.'''
        fraction_to_restart = self.batch_since_restart / (self.steps_per_epoch * self.cycle_length)
        #print(fraction_to_restart)
        if self.shape == "cosine":
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(fraction_to_restart * np.pi))
        else:
            if fraction_to_restart < 0.5:
                lr = fraction_to_restart * (self.max_lr - self.min_lr) / 0.5 + self.min_lr
            else:
                lr = (1 - fraction_to_restart) * (self.max_lr - self.min_lr) / 0.5 + self.min_lr
        self.learning_rates.append(lr)
        return lr

    def on_train_begin(self, logs={}):
        '''Initialize the learning rate to the minimum value at the start of training.'''
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.max_lr)

    def on_batch_end(self, batch, logs={}):
        '''Record previous batch statistics and update the learning rate.'''
        logs = logs or {}
        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        self.batch_since_restart += 1
        K.set_value(self.model.optimizer.lr, self.clr())

    def on_epoch_end(self, epoch, logs={}):
        '''Check for end of current cycle, apply restarts when necessary.'''
        if epoch + 1 == self.next_restart:
            self.batch_since_restart = 0
            self.cycle_length = np.ceil(self.cycle_length * self.mult_factor)
            self.next_restart += self.cycle_length
            self.max_lr *= self.lr_decay
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs={}):
        '''Set weights to the values from the end of the most recent cycle for best performance.'''
        self.model.set_weights(self.best_weights)

class SWA(Callback):

    def __init__(self, epochs_to_train, prop = 0.1, interval = 1,):
        super(SWA, self).__init__()
        self.epochs_to_train = epochs_to_train
        self.prop = prop
        self.interval = interval
        self.n_models = 0
        self.epoch = 0
        
    def on_train_begin(self, logs=None):
        self.nb_epoch = self.params['epochs']
        self.models_weights = []
    def on_epoch_end(self, epoch, logs=None):
        self.epoch += 1
        if epoch % self.interval == 0:
            self.models_weights.append(self.model.get_weights())
            self.n_models += 1
        else:
            pass

    def on_train_end(self, logs=None):
        if self.epoch > 10:
            num_models_to_average = int(np.ceil(self.prop * self.epoch))
#         print(len(self.models_weights))
#         print(len(self.models_weights[0]))
#         print(self.models_weights[0][0].shape)
#         print(self.models_weights[0][1].shape)
            avg_conv_weights = np.mean([weights[0] for weights in self.models_weights[-num_models_to_average:]], axis=0)
            avg_dense_weights = np.mean([weights[1] for weights in self.models_weights[-num_models_to_average:]], axis=0)
        #print(len(avg_conv_weights))
#         print(avg_conv_weights.shape)
#         print(avg_dense_weights.shape)
            self.model.set_weights([avg_conv_weights, avg_dense_weights])

import time

class ProgBar(Callback):
    def __init__(self,
                 num_epochs):
    
        self.num_epochs = num_epochs
        self.start_time = time.time()
        self.stop_time = time.time()
    def on_epoch_start(self, epoch, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.stop_time = time.time()
        time_remaining = ((self.stop_time - self.start_time) * (self.num_epochs - epoch)) // 60
        progress(epoch + 1, self.num_epochs, status='Training Model')
        
    def on_train_end(self, logs=None):
        sys.stdout.write("\n")
    
class OverfitMonitor(Callback):
    def __init__(self, max_delta=0.1, patience=5):
        super(Callback, self).__init__()
        self.max_delta = max_delta
        self.patience = patience
        self.best_loss = 100
        self.overfit = False
       
    def on_train_begin(self, logs=None):
        self.count = 0
        
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('val_loss') < self.best_loss:
            self.best_model_weights = self.model.get_weights()
        if logs.get('val_loss') - logs.get('loss') >= self.max_delta:
            self.count += 1
            #print(self.count)
            if self.count == self.patience:
                print("Overfitting...exiting training")
                self.overfit = True
                self.model.stop_training = True
        else:
            self.count = 0
            
    def on_train_end(self, logs=None):
        if self.overfit:
            print("Setting weights to that of best model")
            self.model.set_weights(self.best_model_weights)

class AntiMotifChecker(Callback):
    def on_train_begin(self, logs={}):
        initial_weights = self.model.get_layer("conv1d_1").get_weights()[0]
        self.num_kernels = initial_weights.shape[2]
        self.kernel_width = initial_weights.shape[0]

    def on_epoch_end(self, epoch, logs={}):
#         if epoch > 80 and epoch < 100:
        if epoch == 80:
            conv_weights = self.model.get_layer("conv1d_1").get_weights()[0]
            #print(conv_weights)
            for i in range(self.num_kernels):
                for w in range(self.kernel_width):
                    if np.max(conv_weights[w,:,i]) < 1:
                        conv_weights[w,:,i] = np.array([0.0,0.0,0.0,0.0])
                        #print("kernel {} position {} set to 0.0".format(i, w))
                    else:
                        continue
                for w in range(self.kernel_width):
                    index = self.kernel_width - w - 1
                    if np.max(conv_weights[index,:,i]) < 1:
                        conv_weights[index,:,i] = np.array([0.0,0.0,0.0,0.0])
                        #print("kernel {} position {} set to 0.0".format(i, index))
                    else:
                        continue
            self.model.get_layer("conv1d_1").set_weights([conv_weights])