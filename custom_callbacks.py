from keras.callbacks import Callback
# from data_generators import DataGeneratorDinucShuffle
from keras import backend as K
import numpy as np

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
            print("Will average last {} models".format(num_models_to_average))
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

class SequentialKernelAddition(Callback):
    def on_train_begin(self, logs={}):
        self.initial_weights = self.model.get_layer("conv1d_1").get_weights()[0]
        self.num_kernels = self.initial_weights.shape[2]
        self.kernel_width = self.initial_weights.shape[0]

        self.current_weights = np.zeros((self.kernel_width, 4, self.num_kernels))
        self.current_weights[:,:,0] = self.initial_weights[:,:,0]
        self.model.get_layer("conv1d_1").set_weights([self.current_weights])
        self.dense_weights = np.zeros((self.num_kernels,1))
        self.dense_weights[0,0] = .01
        self.model.get_layer("dense_1").set_weights([self.dense_weights])
        self.on_kernel = 0
    def on_epoch_end(self, epoch, logs={}):
        if (epoch + 1) % 25 == 0 and self.on_kernel < (self.num_kernels - 1):
            self.on_kernel += 1
            print("Adding kernel {}".format(self.on_kernel+1))
            self.conv_weights = self.model.get_layer("conv1d_1").get_weights()[0]
            self.dense_weights = self.model.get_layer("dense_1").get_weights()[0]
            self.conv_weights[:,:,self.on_kernel] = self.initial_weights[:,:,self.on_kernel]
            self.dense_weights[self.on_kernel,0] = 0.01
            self.model.get_layer("conv1d_1").set_weights([self.conv_weights])
            self.model.get_layer("dense_1").set_weights([self.dense_weights])