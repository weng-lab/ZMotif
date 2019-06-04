import numpy as np
from keras.callbacks import Callback
import threading
import random
from keras.models import Model
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, GlobalMaxPooling1D, Dropout, Input, maximum, LeakyReLU, Lambda, GaussianNoise, concatenate, BatchNormalization, SpatialDropout1D
from keras import initializers
from keras import backend as K
from keras.constraints import non_neg
from keras import regularizers
from pyfaidx import Fasta
from collections import  Counter

def get_consensus_seq(conv_weights):
    print(conv_weights.shape)
    d = np.array(['A', 'C', 'G', 'T'])
    seq = ''.join(d[conv_weights.argmax(axis=1)])
    return seq

def get_decoded_seq(encoded_seq):
    d = np.array(['A', 'C', 'G', 'T'])
    seq = ''.join(d[encoded_seq.argmax(axis=1)])
    return seq

from keras.callbacks import Callback
import keras.backend as K
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
                 mult_factor=2):

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_decay = lr_decay

        self.batch_since_restart = 0
        self.next_restart = cycle_length

        self.steps_per_epoch = steps_per_epoch

        self.cycle_length = cycle_length
        self.mult_factor = mult_factor

        self.history = {}
        self.learning_rates = []

    def clr(self):
        '''Calculate the learning rate.'''
        fraction_to_restart = self.batch_since_restart / (self.steps_per_epoch * self.cycle_length)
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(fraction_to_restart * np.pi))
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


def encode_sequence(seq, N = [0, 0, 0, 0]):
    d = { 'A' : [1, 0, 0, 0],
          'C' : [0, 1, 0, 0],
          'G' : [0, 0, 1, 0],
          'T' : [0, 0, 0, 1],
          'N' : N }

    return np.array([d[nuc] for nuc in list(seq)]).astype('float32')
        
def one_hot_encode_fasta(input_fasta):
    fasta = Fasta(input_fasta, as_raw=True, sequence_always_upper=True)
    seq_list = [encode_sequence(seq[:]) for seq in fasta]
    coords = list(fasta.keys())
    coords_tuple_list = coords_to_tuple(coords)
    return seq_list, coords_tuple_list

def coords_to_tuple(coords):
    coords_tup_list = []
    for coord in coords:
        chrom = coord.split(":")[0]
        start = coord.split(":")[1].split("-")[0]
        stop = coord.split(":")[1].split("-")[1]
        coords_tup_list.append((chrom, start, stop))
    return coords_tup_list
    
def fasta_to_array(input_fasta, pad_by = 0, zero_background = False, return_seq_lens = False):
    fasta = Fasta(input_fasta, as_raw=True, sequence_always_upper=True)
    seq_lens = []
    for seq in fasta:
        seq_lens.append(len(seq))
    seq_lens = np.array(seq_lens)
    max_seq_len = np.max(seq_lens)
    num_seqs = seq_lens.shape[0]
    print(max_seq_len)
    print(num_seqs)

    if zero_background:
        seq_array = np.zeros((num_seqs, max_seq_len + 2 * pad_by, 4))
    else:
        seq_array = 0.25 * np.ones((num_seqs, max_seq_len + 2 * pad_by, 4))
    print(max_seq_len + 2 * pad_by)
    for i, seq in enumerate(fasta):
        #start = pad_by + np.random.randint(0, max_seq_len - seq_lens[i] + kernel_width + 1)
        start = int(max_seq_len / 2 - seq_lens[i] / 2)
        stop = start + seq_lens[i]
        seq_array[i,start:stop,:] = encode_sequence(seq[:])
    if return_seq_lens:
        return seq_array, seq_lens
    else:
        return seq_array

def fasta_to_array_2(input_fasta, aug_by = 100, kernel_width = 40):
    fasta = Fasta(input_fasta, as_raw=True, sequence_always_upper=True)
    seq_lens = []
    for seq in fasta:
        seq_lens.append(len(seq))
    seq_lens = np.array(seq_lens)
    max_seq_len = np.max(seq_lens)
    num_seqs = seq_lens.shape[0]
    print(max_seq_len)
    print(num_seqs)
    
    seq_array = 0.25 * np.ones((num_seqs, max_seq_len + aug_by + 3 * kernel_width, 4))
    for i, seq in enumerate(fasta):
        start = aug_by + kernel_width + np.random.randint(0, max_seq_len - seq_lens[i] + kernel_width + 1)
        #print(start)
        #start = int(max_seq_len / 2 - seq_lens[i] / 2)
        stop = start + seq_lens[i]
        seq_array[i,start:stop,:] = encode_sequence(seq[:])
    return seq_array

def nuc_freqs_from_fasta(input_fasta):
    fasta = Fasta(input_fasta, as_raw=True, sequence_always_upper=True)
    A = 0
    C = 0
    G = 0
    T = 0
    for seq in fasta:
        c = Counter(seq[:])
        A += c['A']
        C += c['C']
        G += c['G']
        T += c['T']
    total_nucs = A + C + G + T
    return np.array([A, C, G, T]) / total_nucs

def get_sample(arr, arr_len, n_iter, sample_size):
    start_idx = (n_iter * sample_size) % arr_len
    if start_idx + sample_size >= arr_len:
        np.random.shuffle(arr)
        start_idx = np.random.randint(0, arr_len - sample_size)
    return arr[start_idx:start_idx+sample_size]

def data_gen(pos_seq_array, neg_seq_array, max_seq_len, batch_size=32, augment_by = 0, background = [0.25, 0.25, 0.25, 0.25]):
    n_iter = 0
    sample_size = batch_size // 2
    labels = np.array([1 for i in range(sample_size)] + [0 for i in range(sample_size)])
    num_pos_seqs = pos_seq_array.shape[0]
    num_neg_seqs = neg_seq_array.shape[0]
    while True:
        pos_sample = get_sample(pos_seq_array, num_pos_seqs, n_iter, sample_size)
        neg_sample = get_sample(neg_seq_array, num_neg_seqs, n_iter, sample_size)
        n_iter += 1
        output = 0.25 * np.ones((batch_size, max_seq_len + augment_by, 4), dtype=np.float32)
        start_indices = np.random.randint(0,augment_by + 1, batch_size)
        #start_index = np.random.randint(0, augment_by + 1)
        for i in range(sample_size):
            output[i, start_indices[i]:start_indices[i] + max_seq_len,:] = pos_sample[i,:,:]
        for i in range(sample_size):
            output[i + sample_size, start_indices[i + sample_size]:(start_indices[i + sample_size] +max_seq_len),:] = neg_sample[i,:,:]
        #output = np.vstack((pos_sample, neg_sample))
        yield output, labels
        #yield np.vstack((pos_sample, neg_sample)), labels

def data_gen_2(seq_array, labels, batch_size=32, aug_by = 100):
    n_iter = 0
    num_samples = seq_array.shape[0]
    while True:
        start_idx = (n_iter * batch_size) % num_samples
        crop_by = np.random.randint(40, aug_by)
        if start_idx + batch_size > num_samples:
            rng_state = np.random.get_state()
            np.random.shuffle(seq_array)
            np.random.set_state(rng_state)
            np.random.shuffle(labels)
            n_iter += 1
            output_array = seq_array[0:batch_size, crop_by:,:]
            output_labels = labels[0:batch_size]
            yield output_array, output_labels
        else:
            output_array = seq_array[start_idx:start_idx + batch_size, crop_by:,:]
            output_labels = labels[start_idx:start_idx + batch_size]
            n_iter += 1
            yield output_array, output_labels
             

class SWA_new(Callback):

    def __init__(self, start_epoch, interval):
        super(SWA_new, self).__init__()
        self.start_epoch = start_epoch
        self.interval = interval
        self.n_models = 0

    def on_train_begin(self, logs=None):
        self.nb_epoch = self.params['epochs']
    def on_epoch_end(self, epoch, logs=None):
        if epoch == self.start_epoch:
            self.swa_weights = self.model.get_weights()
            self.n_models += 1

        elif epoch > self.start_epoch and epoch % self.interval == 0:
            for i, current_layer_weights in enumerate(self.model.get_weights()):
                self.swa_weights[i] = (self.swa_weights[i] * (self.n_models) + current_layer_weights) / (self.n_models + 1)
            self.n_models += 1
        else:
            pass

    def on_train_end(self, logs=None):
        self.model.set_weights(self.swa_weights)
        
def construct_model(num_kernels=8, kernel_width=20, seq_len=None, num_dense=64, dropout_prop=0.0, kernel_initializer='glorot_uniform', optimizer='adam', activation='linear', num_classes=1, l1_reg=0.0000001, gaussian_noise = 0.0, spatial_dropout = 0.0):
    seq_input = Input(shape=(seq_len,4))
    noisy_seq = GaussianNoise(gaussian_noise)(seq_input)
    rc_op = Lambda(lambda x: K.reverse(x,axes=(1,2)))
    noisy_seq_rc = rc_op(noisy_seq)
    shared_conv = Conv1D(num_kernels, kernel_width, strides=1, padding='same', activation=activation, use_bias=False, kernel_initializer=kernel_initializer, kernel_regularizer=regularizers.l1(l1_reg), bias_initializer='zeros')
    conv_for = shared_conv(noisy_seq)
    conv_rc = shared_conv(noisy_seq_rc)
    merged = maximum([conv_for, conv_rc])
    pooled = GlobalMaxPooling1D()(merged)
    #drop = Dropout(0.25)(pooled)
    #dense = Dense(num_dense)(drop1)
    #drop2 = Dropout(dense_dropout)(dense)
    #batch_norm = BatchNormalization()(drop)
    output = Dense(num_classes, activation='sigmoid', use_bias=False, kernel_constraint=non_neg())(pooled)
    model = Model(inputs=seq_input, outputs=output)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model

def construct_classifier(num_kernels=8, kernel_width=20, seq_len=None, num_dense=64, dropout_prop=0.0, kernel_initializer='glorot_uniform', optimizer='adam', activation='linear', num_classes=1):
    seq_input = Input(shape=(seq_len,4))
    noisy_seq = GaussianNoise(0.1)(seq_input)
    rc_op = Lambda(lambda x: K.reverse(x,axes=(1,2)))
    noisy_seq_rc = rc_op(noisy_seq)
    shared_conv = Conv1D(num_kernels, kernel_width, strides=1, padding='same', activation=activation, use_bias=False, kernel_initializer=kernel_initializer, kernel_regularizer=regularizers.l1(0.00000001), bias_initializer='zeros', trainable=False)
    conv_for = shared_conv(noisy_seq)
    conv_rc = shared_conv(noisy_seq_rc)
    merged = maximum([conv_for, conv_rc])
    pooled = GlobalMaxPooling1D()(merged)
    drop1 = Dropout(dropout_prop)(pooled)
    dense1 = Dense(num_dense)(drop1)
    drop2 = Dropout(dropout_prop)(dense1)
    output = Dense(num_classes, activation='sigmoid', use_bias=False, kernel_constraint=non_neg())(drop2)
    model = Model(inputs=seq_input, outputs=output)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model

def get_motif_matches(pos_seq_tup, conv_weights):
    kernel_width = conv_weights.shape[0]
    num_kernels = conv_weights.shape[2]
    input_for = Input(shape=(None,4))
    rc_layer = Lambda(lambda x: K.reverse(x,axes=(1,2)))
    input_rc = rc_layer(input_for)
    shared_conv = Conv1D(1, kernel_width, strides=1, padding='same', activation='linear', use_bias=False, bias_initializer='zeros', name = 'match_conv')
    conv_for = shared_conv(input_for)
    conv_rc = shared_conv(input_rc)
    output = concatenate([conv_for, conv_rc])
    model = Model(inputs=input_for, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    motif_matches = []
    for i in range(num_kernels):
        temp_list = []
        model.get_layer('match_conv').set_weights([conv_weights[:,:,i].reshape((kernel_width, 4, 1))])
        for seq in pos_seq_tup:
            temp_seq = seq[0]
            temp_seq = np.expand_dims(temp_seq, axis=0)
            conv_seq = model.predict(temp_seq)
            max_idx = np.unravel_index(np.argmax(conv_seq, axis=None), conv_seq.shape)
            max_val = np.max(conv_seq)
            if max_idx[2] == 0:
                decoded_seq = get_decoded_seq(temp_seq[0,max_idx[1]-kernel_width//2:max_idx[1]+kernel_width//2,:])
                if len(decoded_seq) == kernel_width:
                    temp_list.append((decoded_seq, max_val,"+", temp_seq.shape[1], max_idx[1], seq[2][0], int(seq[2][1]) + max_idx[1] - kernel_width//2, int(seq[2][1]) + max_idx[1] + kernel_width//2))
            if max_idx[2] == 1:
                temp_seq = temp_seq[:,::-1,::-1]
                decoded_seq = get_decoded_seq(temp_seq[0,max_idx[1]-kernel_width//2:max_idx[1]+kernel_width//2,:])
                if len(decoded_seq) == kernel_width:
                    temp_list.append((decoded_seq, max_val,"-", temp_seq.shape[1], max_idx[1], seq[2][0], int(seq[2][2]) - max_idx[1] - kernel_width//2, int(seq[2][2]) - max_idx[1] + kernel_width//2))
        temp_list.sort(key=lambda x: x[1], reverse=True)
        motif_matches.append(temp_list)
    return motif_matches

def fimo(seq_array, conv_weights):
    kernel_width = conv_weights.shape[0]
    num_kernels = conv_weights.shape[2]
    input_for = Input(shape=(None,4))
    rc_layer = Lambda(lambda x: K.reverse(x,axes=(1,2)))
    input_rc = rc_layer(input_for)
    shared_conv = Conv1D(num_kernels, kernel_width, strides=1, padding='valid', activation='linear', use_bias=False, bias_initializer='zeros', name='fimo_conv')
    conv_for = shared_conv(input_for)
    conv_rc = shared_conv(input_rc)
    merge = maximum([conv_for, conv_rc])
    output = GlobalMaxPooling1D()(merge)
    model = Model(inputs=input_for, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    model.get_layer('fimo_conv').set_weights([conv_weights])
    output_array = model.predict(seq_array)
    return output_array
            

    
def conv_weights_to_meme(weights, output_file, prefix):
    pfms = 100 * 0.25 * 2**weights
    num_kernels = weights.shape[2]
    with open(output_file, "w") as f:
        f.write("MEME version 4 \n\n")
        f.write("ALPHABET= ACGT \n\n")
        f.write("strands: + - \n\n")
        f.write("Background letter frequencies \n")
        f.write("A 0.25 C 0.25 G 0.25 T 0.25 \n\n")

        for i in range(num_kernels):
            f.write("MOTIF " + prefix + "." + str(i) + "\n")
            f.write("letter-probability matrix: alength= 4 w= {} \n".format(str(pfms.shape[0])))
            tempPfm = pfms[:,:,i]
            for j in range(tempPfm.shape[0]):
                tempPpm = tempPfm[j,:] / np.sum(tempPfm[j,:])
                f.write("\t".join([str(x) for x in tempPpm]))
                f.write("\n")
            f.write("\n")

def ppms_to_meme(ppms_tup_list, output_file, prefix):
    with open(output_file, "w") as f:
        f.write("MEME version 4 \n\n")
        f.write("ALPHABET= ACGT \n\n")
        f.write("strands: + - \n\n")
        f.write("Background letter frequencies \n")
        f.write("A 0.25 C 0.25 G 0.25 T 0.25 \n\n")
        for i, ppm_tup in enumerate(ppms_tup_list):
            ppm = ppm_tup[0]
            nsites = ppm_tup[1]
            f.write("MOTIF " + prefix + "." + str(i+1) + "\n")
            f.write("letter-probability matrix: alength= 4 w= {} nsites= {} \n".format(str(ppm.shape[0]), nsites))
            for j in range(ppm.shape[0]):
                tempPpm = ppm[j,:]
                f.write("\t".join([str(x) for x in tempPpm]))
                f.write("\n")
            f.write("\n")
                
def seq_list_to_ppm(seq_list):
    encoded_seq_list = [encode_sequence(seq) for seq in seq_list]
    encoded_seq_array = np.array(encoded_seq_list)
    pfm = np.sum(encoded_seq_array, axis=0)
    ppm = pfm / np.sum(pfm, axis=1).reshape((pfm.shape[0],1))
    return ppm
    
def multitask_loss(y_true, y_pred):
    # Avoid divide by 0
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # Multi-task loss
    return K.mean(K.sum(- y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred), axis=1))


def trim_weights_from_end(conv_weights):
    num_kernels = conv_weights.shape[2]
    kernel_width = conv_weights.shape[0]
    trimmed_weights = np.zeros(conv_weights.shape)
    for i in range(num_kernels):
        kernel = conv_weights[:,:,i]
        # trim from front
        for j in range(kernel_width):
            if np.max(kernel[j,:]) < 0:
                kernel[j,:] = np.zeros((1,4))
            else:
                break
        for j in range(kernel_width):
            if np.max(kernel[-j,:]) < 0:
                kernel[-j,:] = np.zeros((1,4))
            else:
                 break
        trimmed_weights[:,:,i] = kernel
    return(trimmed_weights)
