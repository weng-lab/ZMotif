# Seed value (can actually be different for each attribution step)
seed_value= 0

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
# tf.random.set_seed(seed_value) # tensorflow 2.x
tf.set_random_seed(seed_value) # tensorflow 1.x

# 5. Configure a new global `tensorflow` session
from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=4)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

from pyfaidx import Fasta
import subprocess
import sys
import re
import string 
import time
import json
from subprocess import PIPE, run
from keras.layers import Input, Lambda, Conv1D, GaussianNoise, maximum, GlobalMaxPooling1D, Dense, concatenate, Activation, Add, BatchNormalization, Dropout
from keras import initializers
from keras.constraints import non_neg
from keras import regularizers
from keras.models import Model
from keras.callbacks import Callback, CSVLogger, EarlyStopping
from keras.utils import Sequence
from sklearn.metrics import roc_auc_score, roc_curve
from collections import Counter, OrderedDict
import h5py

def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s %s\r' % (bar, percents, '%', status))
    sys.stdout.flush()
    
DNA_SEQ_DICT = {
    'A' : [1, 0, 0, 0],
    'C' : [0, 1, 0, 0],
    'G' : [0, 0, 1, 0],
    'T' : [0, 0, 0, 1],
}

def encode_sequence(seq, N = [0, 0, 0, 0], seq_dict = None, useN = None):
    if seq_dict is None:
        seq_dict = DNA_SEQ_DICT
    if useN == 'uniform':
        N = [(1/len(seq_dict)) for _ in seq_dict]
    elif useN == 'zeros':
        N = [0 for _ in seq_dict]
    d = { **seq_dict, 'N' : N }
    return np.array([d[nuc] for nuc in list(seq)]).astype('float32')
 
def decode_sequence(encoded_seq, seq_dict = None):
    if seq_dict is None:
        seq_dict = DNA_SEQ_DICT
    seq_list = encoded_seq.astype('int').tolist()
    def decode_base(encoded_base):
        for letter,onehot in seq_dict.items():
            if np.array_equal(encoded_base, onehot):
                return letter
        return "N"
    return "".join(decode_base(b) for b in encoded_seq.astype('int'))

def ppm_to_pwm(ppm, background = [0.25, 0.25, 0.25, 0.25]):
    pwm = np.zeros(ppm.shape)
    w = ppm.shape[0]
    for i in range(w):
        for j in range(4):
            pwm[i,j] = np.log2((ppm[i,j] + .001) / 0.25)
    return pwm

# altschulEriksonDinuclShuffle.py
# P. Clote, Oct 2003

def computeCountAndLists(s):

    #Initialize lists and mono- and dinucleotide dictionaries
    List = {} #List is a dictionary of lists
    List['A'] = []; List['C'] = [];
    List['G'] = []; List['T'] = [];
    # FIXME: is this ok?
    List['N'] = []
    nuclList   = ["A","C","G","T","N"]
    s       = s.upper()
    #s       = s.replace("U","T")
    nuclCnt    = {}  #empty dictionary
    dinuclCnt  = {}  #empty dictionary
    for x in nuclList:
        nuclCnt[x]=0
        dinuclCnt[x]={}
        for y in nuclList:
            dinuclCnt[x][y]=0

    #Compute count and lists
    nuclCnt[s[0]] = 1
    nuclTotal     = 1
    dinuclTotal   = 0
    for i in range(len(s)-1):
        x = s[i]; y = s[i+1]
        List[x].append( y )
        nuclCnt[y] += 1; nuclTotal  += 1
        dinuclCnt[x][y] += 1; dinuclTotal += 1
    assert (nuclTotal==len(s))
    assert (dinuclTotal==len(s)-1)
    return nuclCnt,dinuclCnt,List


def chooseEdge(x,dinuclCnt):
    z = random.random()
    denom=dinuclCnt[x]['A']+dinuclCnt[x]['C']+dinuclCnt[x]['G']+dinuclCnt[x]['T']+dinuclCnt[x]['N']
    numerator = dinuclCnt[x]['A']
    if z < float(numerator)/float(denom):
        dinuclCnt[x]['A'] -= 1
        return 'A'
    numerator += dinuclCnt[x]['C']
    if z < float(numerator)/float(denom):
        dinuclCnt[x]['C'] -= 1
        return 'C'
    numerator += dinuclCnt[x]['G']
    if z < float(numerator)/float(denom):
        dinuclCnt[x]['G'] -= 1
        return 'G'
    numerator += dinuclCnt[x]['T']
    if z < float(numerator)/float(denom):
        dinuclCnt[x]['T'] -= 1
        return 'T'
    dinuclCnt[x]['N'] -= 1
    return 'N'

def connectedToLast(edgeList,nuclList,lastCh):
    D = {}
    for x in nuclList: D[x]=0
    for edge in edgeList:
        a = edge[0]; b = edge[1]
        if b==lastCh: D[a]=1
    for i in range(3):
        for edge in edgeList:
            a = edge[0]; b = edge[1]
            if D[b]==1: D[a]=1
    ok = 0
    for x in nuclList:
        if x!=lastCh and D[x]==0: return 0
    return 1

def eulerian(s):
    nuclCnt,dinuclCnt,List = computeCountAndLists(s)
    #compute nucleotides appearing in s
    nuclList = []
    for x in ["A","C","G","T","N"]:
        if x in s: nuclList.append(x)
    #create dinucleotide shuffle L
    firstCh = s[0]  #start with first letter of s
    lastCh  = s[-1]
    edgeList = []
    for x in nuclList:
        if x!= lastCh: edgeList.append( [x,chooseEdge(x,dinuclCnt)] )
    ok = connectedToLast(edgeList,nuclList,lastCh)
    return ok,edgeList,nuclList,lastCh


def shuffleEdgeList(L):
    n = len(L); barrier = n
    for i in range(n-1):
        z = int(random.random() * barrier)
        tmp = L[z]
        L[z]= L[barrier-1]
        L[barrier-1] = tmp
        barrier -= 1
    return L

def dinuclShuffle(s):
    ok = 0
    while not ok:
        ok,edgeList,nuclList,lastCh = eulerian(s)
    nuclCnt,dinuclCnt,List = computeCountAndLists(s)

    #remove last edges from each vertex list, shuffle, then add back
    #the removed edges at end of vertex lists.
    for [x,y] in edgeList: List[x].remove(y)
    for x in nuclList: shuffleEdgeList(List[x])
    for [x,y] in edgeList: List[x].append(y)

    #construct the eulerian path
    L = [s[0]]; prevCh = s[0]
    for i in range(len(s)-2):
        ch = List[prevCh][0]
        L.append( ch )
        del List[prevCh][0]
        prevCh = ch
    L.append(s[-1])
    #t = string.join(L,"")
    t = "".join(L)
    return t

def fasta_to_seq_list(fasta, store_coords=True):
    if store_coords:
        seq_list = []
        for seq in Fasta(fasta, sequence_always_upper=True, as_raw=True):
            seq_list.append([seq[:], 1.0, seq.name])
    else:
        seq_list = [[seq[:], 1.0] for seq in Fasta(fasta, sequence_always_upper=True, as_raw=True)]
    return(seq_list)


def bg_to_seq_list(bg, genome_fasta):
    genome = Fasta(genome_fasta, sequence_always_upper=True, as_raw=True)
    intervals = []
    with open(bg) as f:
        for line in f:
            split = line.strip().split()
            chrom, start, stop, weight = split
            start = int(start)
            stop = int(stop)
            weight = float(weight)
            intervals.append([chrom, start, stop, weight])
    
    max_weight = np.max([x[3] for x in intervals])
    print(max_weight)
    
    pos_seqs = []
    for interval in intervals:
        chrom, start, stop, weight = interval
        seq = genome[chrom][start:stop]
#         weight = 10 * (weight / max_weight)
        pos_seqs.append([seq[:], weight, chrom + ":" + str(start) + "-" + str(stop)])
        
    pos_seqs.sort(key = lambda x: x[1], reverse=True)
    
    neg_seqs = []
    for seq in pos_seqs:
        neg_seqs.append([dinuclShuffle(seq[0]), 1.0, seq[2] + "_shuffle"])
    
    
    
    return(pos_seqs, neg_seqs)
def get_enriched_kmers(pos_fasta, neg_fasta, n = 1):
    k = 6
    cmd = ["jellyfish", "count", "-m", str(k), "-s", "100M", "-t", "1", pos_fasta]
    run(cmd)
    with open("pos_kmer_counts.fa", "w") as f:
        run(["jellyfish", "dump", "mer_counts.jf"], stdout=f)

    pos_counter = Counter()
    with open("pos_kmer_counts.fa") as f:
        for i, line in enumerate(f):
            if i % 2 == 0:
                n = int(line.strip().split()[0][1:])
            else:
                kmer = line.strip().split()[0]
                pos_counter[kmer] = n

    cmd = ["jellyfish", "count", "-m", str(k), "-s", "100M", "-t", "1", neg_fasta]
    run(cmd)
    with open("neg_kmer_counts.fa", "w") as f:
        run(["jellyfish", "dump", "mer_counts.jf"], stdout=f)

    neg_counter = Counter()
    with open("neg_kmer_counts.fa") as f:
        for i, line in enumerate(f):
            if i % 2 == 0:
                n = int(line.strip().split()[0][1:])
            else:
                kmer = line.strip().split()[0]
                neg_counter[kmer] = n

    n_p_tot = sum(pos_counter.values())
    n_n_tot = sum(neg_counter.values())
    
    results = {}
    for kmer in pos_counter:
        n_p_k = pos_counter[kmer]
        n_n_k = neg_counter[kmer]
        p_p_k = n_p_k / n_p_tot
        p_n_k = n_n_k / n_n_tot

        pi = (n_p_k + n_n_k) / (n_p_tot + n_n_tot)
        try:
            z = (p_p_k - p_n_k) / np.sqrt(pi * (1-pi) * ((1/n_p_k) + (1/n_n_k)))
        except:
            z = -1
        results[kmer] = z
                  
            
        
    sorted_results = {}
    for kmer in sorted(results, key=results.get, reverse=True):
        sorted_results[kmer] = results[kmer]
        
    with open("enriched_kmers.txt", "w") as f:
        for i, kmer in enumerate(sorted_results):
            print(kmer, sorted_results[kmer], sep="\t", file=f)
            if i == 20:
                break
    return(sorted_results)

class DataGeneratorSeqList(Sequence):
    def __init__(self,
                 pos_seqs,
                 neg_seqs=None,
                 batch_size=32,
                 pad_by=24,
                 sort_seqs_on_start=False,
                 reshuffle=False,
                 redraw=False,
                 chrom_sizes=None,
                 genome=None):
        
        
        self.pos_seqs = pos_seqs
        self.reshuffle = reshuffle
        self.redraw = redraw
        self.chrom_sizes = chrom_sizes
        self.genome = genome
        if sort_seqs_on_start == True:
            print("Sorting sequences")
            print(self.pos_seqs[0])
            self.pos_seqs.sort(reverse=True, key = lambda seq: seq[1])
            print(self.pos_seqs[0])
            
        self.neg_seqs = neg_seqs
        if self.neg_seqs:
            self.max_seq_len = np.max([len(seq[0]) for seq in self.pos_seqs + self.neg_seqs])
        else:
            self.max_seq_len = np.max([len(seq[0]) for seq in self.pos_seqs])
            
        self.batch_size = batch_size
        self.b2 = self.batch_size // 2
        self.num_pos = len(self.pos_seqs)
        # Number of iterations after which to shuffle positive sequences
        self.shuffle_pos_every = self.num_pos // self.b2
        if self.shuffle_pos_every == 0:
            self.shuffle_pos_every = 1
        print("Will shuffle positive sequences every {} iterations".format(self.shuffle_pos_every))
        if self.neg_seqs is not None:
            self.num_neg = len(self.neg_seqs)
            self.shuffle_neg_every = self.num_neg // self.b2
            print("Will shuffle negative sequences every {} iterations".format(self.shuffle_neg_every))
        else:
            self.shuffle_neg_every = None
            
            
        self.labels = np.array([1 for i in range(self.b2)] + [0 for i in range(self.b2)])
        
        self.pad_by = pad_by
        self.n_iter = 0
        if self.shuffle_neg_every is not None:
            if self.shuffle_neg_every == 0:
                self.shuffle_neg_every = 1
        
    def __len__(self):
        return(int(np.floor(5000//self.batch_size)))
    
    def __getitem__(self, index):
        output = 0.25*np.ones((self.batch_size, 2*self.pad_by + self.max_seq_len, 4))
        
        self.n_iter += 1
        if self.n_iter % self.shuffle_pos_every == 0:
            random.shuffle(self.pos_seqs)
            
        pos_idx = (self.n_iter * self.b2) % (self.num_pos - self.b2)
        pos_sample = self.pos_seqs[pos_idx:pos_idx + self.b2]
        
        if self.neg_seqs is None or self.reshuffle:
            neg_sample = [(dinuclShuffle(x[0]), 1.0) for x in pos_sample]
        else:
            if self.n_iter % self.shuffle_neg_every == 0:
                if self.redraw:
                    with open("neg.bed", "w") as f:
                        shuffle = subprocess.Popen(["bedtools", "shuffle", "-i", "pos.bed", "-g", self.chrom_sizes, "-incl", "flank.bed"], stdout=subprocess.PIPE)
                        sort = subprocess.Popen(["sort", "-k1,1", "-k2,2n"],
                                                stdin=shuffle.stdout,
                                                stdout=subprocess.PIPE)
                        uniq = subprocess.Popen(["uniq"],
                                                stdin=sort.stdout,
                                                stdout=subprocess.PIPE)
                        subprocess.run(["shuf"], stdin=uniq.stdout, stdout=f)
                        
                        
    
                    with open("neg.fasta", "w") as f:
                        subprocess.run(["bedtools", "getfasta", "-fi", self.genome, "-bed", "neg.bed"], stdout=f)
        
                    self.neg_seqs = fasta_to_seq_list("neg.fasta")
                
                else:
                    random.shuffle(self.neg_seqs)
                
            neg_idx = (self.n_iter * self.b2) % (self.num_neg - self.b2)
            neg_sample = self.neg_seqs[neg_idx:neg_idx+self.b2]
        
        weights = []
        for i, seq in enumerate(pos_sample):
            try:
                seq_len = len(seq[0])
                start = (self.max_seq_len - seq_len) // 2 + self.pad_by
                stop = start + seq_len
                output[i,start:stop,:] = encode_sequence(seq[0], N = [0.25, 0.25, 0.25, 0.25])
                weights.append(seq[1])
            except:
                print(seq)
                quit()
            
        for i, seq in enumerate(neg_sample):
            seq_len = len(seq[0])
            start = (self.max_seq_len - seq_len) // 2 + self.pad_by
            stop = start + seq_len
            output[i+self.b2,start:stop,:] = encode_sequence(seq[0], N = [0.25, 0.25, 0.25, 0.25])
            weights.append(1.0)
            
            
        return((output, self.labels, np.array(weights)))
    
def construct_model(num_kernels=1,
                    kernel_width=24,
                    seq_len=None,
                    dropout_prop=0.0,
                    kernel_initializer=initializers.RandomNormal(stddev=0.001, seed=12),
                    optimizer='adam',
                    activation='linear',
                    num_classes=1,
                    l1_reg=0.0,
                    l2_reg= 0.0,
                    gaussian_noise = 0.0,
                    spatial_dropout = 0.0,
                    rc = True,
                    padding="same",
                    conv_name="shared_conv"):
    if rc:
        seq_input = Input(shape=(seq_len,4))
        rc_op = Lambda(lambda x: K.reverse(x,axes=(1,2)))
        seq_rc = rc_op(seq_input)
        if gaussian_noise > 0.0:
            noisy_seq = GaussianNoise(gaussian_noise)(seq_input)
            noisy_seq_rc = rc_op(noisy_seq)
        
        shared_conv = Conv1D(num_kernels, kernel_width,
                             strides=1, padding=padding, 
                             activation=activation,
                             use_bias=False,
                             kernel_initializer=kernel_initializer,
                             kernel_regularizer=regularizers.l1_l2(l1=l1_reg,
                                                                   l2=l2_reg),
                             bias_initializer='zeros',
                             name=conv_name)

        if gaussian_noise > 0:
            conv_for = shared_conv(noisy_seq)
            conv_rc = shared_conv(noisy_seq_rc)
        else:
            conv_for = shared_conv(seq_input)
            conv_rc = shared_conv(seq_rc)
            

        merged = maximum([conv_for, conv_rc])
        pooled = GlobalMaxPooling1D()(merged)
        if dropout_prop > 0.0:
            dropout = Dropout(dropout_prop)(pooled)
            output = Dense(1, activation='sigmoid',
                       use_bias=False,
                       kernel_initializer=initializers.RandomUniform(minval=0.0, maxval=0.001, seed=12), 
                       kernel_constraint=non_neg(), 
                       bias_initializer='zeros',
                       name="dense_1")(dropout)
        else:
            output = Dense(1, activation='sigmoid',
                           use_bias=False,
                           kernel_initializer=initializers.RandomUniform(minval=0.0, maxval=0.001, seed=12), 
                           kernel_constraint=non_neg(), 
                           bias_initializer='zeros',
                           name="dense_1")(pooled)
        model = Model(inputs=seq_input, outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        return model       
 
def construct_scan_model(conv_weights):
    kernel_width = conv_weights.shape[0]
    num_kernels = conv_weights.shape[2]
    seq = Input(shape=(None,4))
    conv = Conv1D(num_kernels, kernel_width, 
                  name = 'scan_conv',
                  strides=1, 
                  padding='valid', 
                  activation='linear', 
                  use_bias=False, 
                  kernel_initializer='zeros', 
                  bias_initializer='zeros',
                  trainable=False)
    
    conv_seq = conv(seq)
    
    
    model = Model(inputs=seq, outputs=conv_seq)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.get_layer('scan_conv').set_weights([conv_weights])
    return model

def construct_lr_model(conv_weights):
    kernel_width = conv_weights.shape[0]
    num_kernels = conv_weights.shape[2]
    seq_input = Input(shape=(None,4))
    rc_op = Lambda(lambda x: K.reverse(x,axes=(1,2)))
    seq_rc = rc_op(seq_input)
    conv = Conv1D(num_kernels, kernel_width, 
                  name = 'shared_conv',
                  strides=1, 
                  padding='valid', 
                  activation='linear', 
                  use_bias=False, 
                  kernel_initializer='zeros', 
                  bias_initializer='zeros',
                  trainable=False)
    
    conv_for = conv(seq_input)
    conv_rc = conv(seq_rc)
    
    merged = maximum([conv_for, conv_rc])
    pooled = GlobalMaxPooling1D()(merged)
    output = Dense(1, activation='sigmoid', name="output")(pooled)
    
    
    model = Model(inputs=seq_input, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.get_layer('shared_conv').set_weights([conv_weights])
    return model



class SWA(Callback):

    def __init__(self, epochs_to_train, prop = 0.2, interval = 1):
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
            if num_models_to_average < 1:
                print("Taking last model")
                num_models_to_average = 1
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
        if self.shape == "cosine":
            K.set_value(self.model.optimizer.lr, self.max_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.min_lr)

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

class ProgBar(Callback):
    def __init__(self,
                 num_epochs):
    
        self.num_epochs = num_epochs
        self.start_time = time.time()
        self.stop_time = time.time()
    def on_epoch_start(self, epoch, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.stop_time = time.time()
        results = [logs['loss'], logs['val_loss'], logs['acc'], logs['val_acc']]
#         results = [logs['loss'], logs['acc']]
        time_remaining = ((self.stop_time - self.start_time) * (self.num_epochs - epoch)) // 60
        progress(epoch + 1, self.num_epochs, status="train_acc: {0:.2f}; val_acc: {1:.2f}".format(logs['acc'], logs['val_acc']))
        
        
    def on_train_end(self, logs=None):
        sys.stdout.write("\n")    
        
class Giffer(Callback):

    def __init__(self):
        super(Giffer, self).__init__()
        self.iter = 0
        self.weights = []
        self.accs = []
        
    def on_batch_end(self, epoch, logs={}):
        self.weights.append(self.model.get_layer("shared_conv").get_weights()[0])
        self.accs.append(logs['acc'])
    def on_train_end(self, logs={}):
        print(len(self.weights))
        print(len(self.accs))
        gif_data = {}
        for i, (weight, acc) in enumerate(zip(self.weights, self.accs)):
            gif_data[i] = {"weight" : weight.tolist(), "acc" : str(acc)}
            
        with open('gif_data.json', 'w') as f:
            json.dump(gif_data, f)
            
        
def ppm_to_pwm(ppm, background = [0.25, 0.25, 0.25, 0.25]):
    pwm = np.zeros(ppm.shape)
    w = ppm.shape[0]
    for i in range(w):
        for j in range(4):
            pwm[i,j] = np.log2((ppm[i,j] + .001) / 0.25)
    return pwm

        

def ppms_to_meme(ppms, output_file):
    with open(output_file, "w") as f:
        f.write("MEME version 5 \n\n")
        f.write("ALPHABET= ACGT \n\n")
        f.write("strands: + - \n\n")
        f.write("Background letter frequencies \n")
        f.write("A 0.25 C 0.25 G 0.25 T 0.25 \n\n")
        for ppm_tup in ppms:
            motif_id, n_sites, ppm = ppm_tup
            f.write("MOTIF " + motif_id + "\n")
            f.write("letter-probability matrix: alength= 4 w= {} nsites= {} \n".format(str(ppm.shape[0]), n_sites))
            for j in range(ppm.shape[0]):
                tempPpm = ppm[j,:]
                f.write("\t".join([str(x) for x in tempPpm]))
                f.write("\n")
            f.write("\n")
            
def scan_seqs(seq_list, conv_weights, output_prefix,
               mode='anr',
               e = 4,
               thresh="zero"):
    
    bed_file = output_prefix + ".bed"
    if os.path.exists(bed_file):
        os.remove(bed_file)
        
    num_kernels = conv_weights.shape[2]
    w = conv_weights.shape[0]
    
    if thresh == "half-max":
        thresholds = {}
        for i in range(num_kernels):
            max_val = np.sum(np.max(conv_weights[:,:,0], axis=1))
            thresholds[i] = 0.5 * max_val
    
            
    
    scan_model = construct_scan_model(conv_weights)
    
    pfms = [np.zeros((w+2*e,4)) for i in range(num_kernels)]
    n_instances = {i : 0 for i in range(num_kernels)}
    
    
    with open(bed_file, "w") as f:
        for i, [seq, weight, coord] in enumerate(seq_list):

            if i % 1000 == 0:
                # update progress bar
                progress(i, len(seq_list), "Scanning sequences")

            start = int(float(coord.split(":")[1].split("-")[0]))
            stop = int(float(coord.split(":")[1].split("-")[1]))
            chrom = coord.split(":")[0]

            encoded_seq = np.vstack((0.25*np.ones((w,4)), encode_sequence(seq), 0.25*np.ones((w,4))))
            encoded_seq_rc = encoded_seq[::-1,::-1]

            conv_for = scan_model.predict(np.expand_dims(encoded_seq, axis = 0))[0]
            conv_rc = scan_model.predict(np.expand_dims(encoded_seq_rc, axis = 0))[0]

            for k in range(num_kernels):
                if mode == 'anr':
                    matches_for = np.argwhere(conv_for[:,k] > 0)[:,0].tolist()
                    matches_rc = np.argwhere(conv_rc[:,k] > 0)[:,0].tolist()
                    for x in matches_for:
                        motif_start = start + x - w - e
                        motif_end = motif_start + w + e
                        score = conv_for[x,k]
                        pfms[k] += encoded_seq[x-e:x+w+e,:]
                        matched_seq = decode_sequence(encoded_seq[x-e:x+w+e,:])
                        n_instances[k] += 1
                        instance_id = output_prefix + "_" + str(k) + "_" + str(n_instances[k])
                        to_write = "\t".join([str(x) for x in [chrom, motif_start, motif_end, instance_id, score, "+", matched_seq]])
                        f.write(to_write + "\n")

                    for x in matches_rc:
                        motif_end = stop - x + w + e
                        motif_start = motif_end - w - e 
                        score = conv_rc[x,k] 
                        pfms[k] += encoded_seq_rc[x-e:x+w+e,:]
                        matched_seq = decode_sequence(encoded_seq_rc[x-e:x+w+e])
                        n_instances[k] += 1
                        instance_id = output_prefix + "_" + str(k) + "_" + str(n_instances[k])
                        to_write = "\t".join([str(x) for x in [chrom, motif_start, motif_end, instance_id, score, "-", matched_seq]])
                        f.write(to_write + "\n")
                        
                else:
                    max_for = np.max(conv_for[:,k])
                    max_rc = np.max(conv_rc[:,k])
                    if np.max([max_for, max_rc]) > 0:
                        if max_for > max_rc:
                            x = np.argmax(conv_for[:,k])
                            motif_start = start + x - w - e
                            motif_end = motif_start + w + e
                            score = conv_for[x,k]
                            pfms[k] += encoded_seq[x-e:x+w+e,:]
                            matched_seq = decode_sequence(encoded_seq[x-e:x+w+e,:])
                            n_instances[k] += 1
                            instance_id = output_prefix + "_" + str(k) + "_" + str(n_instances[k])
                            to_write = "\t".join([str(x) for x in [chrom, motif_start, motif_end, instance_id, score, "+", matched_seq]])
                            f.write(to_write + "\n")
                            
                        else:
                            x = np.argmax(conv_rc[:,k])
                            motif_end = stop - x + w + e
                            motif_start = motif_end - w - e 
                            score = conv_rc[x,k] 
                            pfms[k] += encoded_seq_rc[x-e:x+w+e,:]
                            matched_seq = decode_sequence(encoded_seq_rc[x-e:x+w+e])
                            n_instances[k] += 1
                            instance_id = output_prefix + "_" + str(k) + "_" + str(n_instances[k])
                            to_write = "\t".join([str(x) for x in [chrom, motif_start, motif_end, instance_id, score, "-", matched_seq]])
                            f.write(to_write + "\n")
                    
    
    ppms = []
    for i, pfm in enumerate(pfms):
        n = n_instances[i]
        if n >= 100:
            try:
                ppm = pfm/pfm.sum(axis=1, keepdims=True)
                motif_id = output_prefix + "_" + str(i)
                ppms.append((motif_id, n, ppm))
            except:
                pass
        
    ppms_to_meme(ppms, output_prefix + ".meme")
    return(ppms)

# def lr(ppms, motif_ids, train_generator, test_generator):
#     n = len(ppms)
#     w = np.max([ppm.shape[0] for ppm in ppms])
#     conv_weights = np.zeros((w,4,n))
    
#     for i, ppm in enumerate(ppms):
#         pwm = ppm_to_pwm(ppm)
#         start = (w - ppm.shape[0]) // 2
#         stop = start + ppm.shape[0]
#         conv_weights[start:stop,:,i] = pwm
        
#     model = construct_lr_model(conv_weights)
    
#     progbar = ProgBar(100)
    
#     early_stopping = EarlyStopping(monitor='val_loss',
#                                    patience=5,
#                                    verbose=0,
#                                    mode='auto',
#                                    baseline=None,
#                                    restore_best_weights=False)
    
# #     giffer = Giffer()
#     callbacks_list = [progbar, early_stopping]
    
#     model.fit_generator(train_generator,
#                         steps_per_epoch=15, 
#                         epochs=100,
#                         verbose=0,
#                         callbacks=callbacks_list,
#                         validation_data=test_generator,
#                         validation_steps=2)

#     lr_weights = np.ravel(model.get_layer("output").get_weights()[0])
    
#     ranked = np.argsort(lr_weights)
#     for i in ranked[::-1][:16]:
#         print(motif_ids[i], lr_weights[i])
        
#     dense_weights = model.get_layer("output").get_weights()[0]
#     return(conv_weights[:,:,ranked[::-1][:16]], dense_weights[ranked[::-1][:16],:])

def draw_seqs_from_genome(pos_fasta, genome, chrom_sizes, flank=10000):
    seqs = Fasta(pos_fasta, sequence_always_upper=True, as_raw=True)
    neg_fasta = "neg.fasta"
    with open("pos.bed", "w") as f:
        for seq in seqs:
            coord = seq.name
            chrom = coord.split(":")[0]
            start = coord.split(":")[1].split("-")[0]
            stop = coord.split(":")[1].split("-")[1]
            print(chrom, start, stop, file=f, sep="\t")
    
    with open("flank.bed", "w") as f:
        subprocess.run(["bedtools", "flank", "-b", str(flank), "-i", "pos.bed", "-g", chrom_sizes], stdout=f)
        
    with open("neg.bed", "w") as f:
        shuffle = subprocess.Popen(["bedtools", "shuffle", "-i", "pos.bed", "-g", chrom_sizes, "-incl", "flank.bed"], stdout=subprocess.PIPE)
        sort = subprocess.Popen(["sort", "-k1,1", "-k2,2n"],
                                stdin=shuffle.stdout,
                                stdout=subprocess.PIPE)
        uniq = subprocess.Popen(["uniq"],
                                stdin=sort.stdout,
                                stdout=subprocess.PIPE)
        subprocess.run(["shuf"], stdin=uniq.stdout, stdout=f)
                        
    
    with open(neg_fasta, "w") as f:
        subprocess.run(["bedtools", "getfasta", "-fi", genome, "-bed", "neg.bed"], stdout=f)
        
    neg_seqs = fasta_to_seq_list(neg_fasta)
    return(neg_seqs)
    
        
def construct_argument_parser():
    import argparse
    parser = argparse.ArgumentParser(description='ZMotif')
    parser.add_argument('-pos_fasta', '--pos_fasta', help='Fasta file containing positive sequences', type=str, required=False, default=None)
    parser.add_argument('-bg', '--bedGraph', help='BedGraph file containing regions and signal / weights', type=str, required=False, default=None)
    parser.add_argument('-bed', '--bed', help='Bed file containing regions of interest', type=str, required=False, default=None)
    parser.add_argument('-chrom_sizes', '--chrom_sizes', help='Chromosome sizes', type=str, required=False, default=None)
    parser.add_argument('-g', '--genome', help='Genome FASTA file', type=str, required=False, default=None)
    parser.add_argument('-flank', '--flank', help='Length of adjacent regions to draw negative regions from (used as "-b" flag in bedtools flank', type=int, required=False, default=10000)
    
    parser.add_argument('-motif_db', '--motif_db', help='Motif databse to compare found motifs', type=str, required=True)
    parser.add_argument('-neg_fasta', '--neg_fasta', help='Fasta file containing negative sequences', type=str, required=False)
    parser.add_argument('-o', '--output_prefix', help='Prefix of output files', type=str, required=False, default="zmotif")
    parser.add_argument('-e', '--epochs', help='Number of epochs', type=int, required=False, default=100)
    parser.add_argument('-w', '--width', help='Maximum motif width', type=int, required=False, default=24)
    parser.add_argument('-n', '--n_motifs', help='Maximum number of motifs', type=int, required=False, default=16)
    parser.add_argument('-seed', '--seed_motif', help='Seed motif in meme format', type=str, required=False, default=None)
    parser.add_argument('-seed_db', '--seed_database', help='Motifs to seed network with', type=str, required=False, default=None)
    parser.add_argument('-b', '--batch_size', help='Batch size', type=int, required=False, default=32)
    parser.add_argument('-noise', '--gaussian_noise', help='Gaussian noise', type=float, required=False, default=0.0)
    parser.add_argument('-l1', '--l1_reg', help='L1 regularization', type=float, required=False, default=0.0)
    parser.add_argument('-l2', '--l2_reg', help='L2 regularization', type=float, required=False, default=0.0)
    parser.add_argument('-dropout', '--dropout', help='Convolution dropout proportion', type=float, required=False, default=0.0)
    parser.add_argument('-seed_weights', '--seed_weights', help='Seed weights', type=str, required=False, default=None)
    return parser

def main():
    args = construct_argument_parser().parse_args()
    pos_fasta = args.pos_fasta
    bg = args.bedGraph
    genome = args.genome
    chrom_sizes = args.chrom_sizes
    flank = args.flank
    neg_fasta = args.neg_fasta
    motif_db = args.motif_db
    output_prefix = args.output_prefix
    n_motifs = args.n_motifs
    epochs = args.epochs
    seed_motif = args.seed_motif
    seed_db = args.seed_database
    w = args.width
    batch_size = args.batch_size
    gn = args.gaussian_noise
    l1 = args.l1_reg
    l2 = args.l2_reg
    dropout = args.dropout
    seed_weights = args.seed_weights
    
    if pos_fasta:
        pos_seqs = fasta_to_seq_list(pos_fasta)
        
    if neg_fasta:
        neg_seqs = fasta_to_seq_list(neg_fasta)
        redraw=False
    else:
        neg_seqs = draw_seqs_from_genome(pos_fasta, genome, chrom_sizes, flank=flank)
        neg_fasta = "neg.fasta"
        redraw = True
    
    if seed_motif is None:
        enriched_kmers = get_enriched_kmers(pos_fasta, neg_fasta)
    
        
    
    train_test_split = 0.9
    
    num_pos = len(pos_seqs)
    pos_split_index = int(train_test_split*num_pos)
    random.shuffle(pos_seqs)
    train_pos_seqs = pos_seqs[:pos_split_index]
    test_pos_seqs = pos_seqs[pos_split_index:]
    
    num_neg = len(neg_seqs)
    neg_split_index = int(train_test_split*num_neg)
    train_neg_seqs = neg_seqs[:neg_split_index]
    test_neg_seqs = neg_seqs[neg_split_index:]
    
    steps_per_epoch = int(np.floor(20000//batch_size))
    validation_steps = int(np.floor(5000//batch_size))
    
    train_data_generator = DataGeneratorSeqList(train_pos_seqs,
                                                neg_seqs = train_neg_seqs,
                                                batch_size=batch_size,
                                                sort_seqs_on_start=False,
                                                redraw=redraw, chrom_sizes=chrom_sizes, genome=genome)
    
    test_data_generator = DataGeneratorSeqList(test_pos_seqs,
                                               neg_seqs = test_neg_seqs,
                                               batch_size=batch_size)
        
    
    
    if seed_motif is None:
        model = construct_model(num_kernels=1,
                                kernel_width=w,
                                seq_len=None,
                                optimizer='adam',
                                activation='linear',
                                l1_reg=0.0,
                                gaussian_noise = 0.0,
                                rc = True,
                                conv_name="shared_conv")

        conv_weights = model.get_layer("shared_conv").get_weights()[0]

        start_idx = (w - 6) // 2
        stop_idx = start_idx + 6

        for i, kmer in enumerate(enriched_kmers):
            print(kmer)
            encoded_seed = encode_sequence(kmer)
            conv_weights[start_idx:stop_idx,:,i] = encoded_seed
            break
            
    else:
        ppm = []
        with open (seed_motif) as meme:
            lines = [line.strip().split() for line in meme.readlines()]
        
        lines = [line for line in lines if len(line) > 0]
        for line in lines:
            if re.match(r'^-?\d+(?:\.\d+)$', line[0]) is not None:
                ppm.append([float(x) for x in line])
                
        ppm = np.array(ppm)
        pwm = ppm_to_pwm(ppm)
        w = pwm.shape[0]
        print("Setting kernel width to {}".format(w))
        model = construct_model(num_kernels=1,
                                kernel_width=w,
                                seq_len=None,
                                optimizer='adam',
                                activation='linear',
                                l1_reg=0.0,
                                gaussian_noise = 0.0,
                                rc = True,
                                conv_name="shared_conv")

        conv_weights = model.get_layer("shared_conv").get_weights()[0]
        conv_weights[:,:,0] = pwm
            
    model.get_layer("shared_conv").set_weights([conv_weights])
    
    schedule = SGDRScheduler(min_lr=0.01,
                             max_lr=0.1,
                             steps_per_epoch=steps_per_epoch,
                             lr_decay=1.0,
                             cycle_length=1,
                             mult_factor=1.0, 
                             shape="triangular")


    progbar = ProgBar(100)

    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=5,
                                   verbose=0,
                                   mode='auto',
                                   baseline=None,
                                   restore_best_weights=False)

#     giffer = Giffer()
    callbacks_list = [schedule, progbar]

    model.fit_generator(train_data_generator,
                        steps_per_epoch=steps_per_epoch, 
                        epochs=10,
                        verbose=0,
                        callbacks=callbacks_list,
                        validation_data=test_data_generator,
                        validation_steps=validation_steps)

    seed_conv_weights = model.get_layer("shared_conv").get_weights()[0]
    seed_dense_weights = model.get_layer("dense_1").get_weights()[0]
    
    model = construct_model(num_kernels=n_motifs,
                            kernel_width=w,
                            seq_len=None,
                            optimizer='adam',
                            activation='linear',
                            l1_reg=l1,
                            l2_reg=l2,
                            dropout_prop=dropout,
                            gaussian_noise = gn,
                            rc = True,
                            conv_name="shared_conv")
    
    conv_weights = model.get_layer("shared_conv").get_weights()[0]
    dense_weights = model.get_layer("dense_1").get_weights()[0]
    
    conv_weights[:,:,0] = seed_conv_weights[:,:,0]
    dense_weights[0] = seed_dense_weights
    
    model.get_layer("shared_conv").set_weights([conv_weights])
    model.get_layer("dense_1").set_weights([dense_weights])
    
    if seed_weights is  not None:
        f = h5py.File(seed_weights, 'r')
        conv_weights = f['shared_conv']['shared_conv_2']['kernel:0']
        dense_weights = f["dense_1"]["dense_1_1"]['kernel:0']
        model.get_layer("shared_conv").set_weights([conv_weights])
        model.get_layer("dense_1").set_weights([dense_weights])
        f.close()
    train_data_generator = DataGeneratorSeqList(train_pos_seqs,
                                                neg_seqs = train_neg_seqs,
                                                batch_size=batch_size,
                                                sort_seqs_on_start=False,
                                                reshuffle=False,
                                                redraw=redraw, chrom_sizes=chrom_sizes, genome=genome)
    
#     train_data_generator = DataGeneratorSeqList(train_pos_seqs,
#                                                 neg_seqs = None,
#                                                 batch_size=batch_size,
#                                                 sort_seqs_on_start=False,
#                                                 reshuffle=False,
#                                                 redraw=True, chrom_sizes=chrom_sizes, genome=genome)
    
    
    schedule = SGDRScheduler(min_lr=0.01,
                             max_lr=0.01,
                             steps_per_epoch=steps_per_epoch,
                             lr_decay=1.0,
                             cycle_length=1,
                             mult_factor=1.0, 
                             shape="triangular")
    
    progbar = ProgBar(epochs)
    swa = SWA(epochs, prop = 0.2, interval = 1)
#     giffer = Giffer()
    callbacks_list = [schedule, progbar, swa]
    
    
    history = model.fit_generator(train_data_generator,
                        steps_per_epoch=steps_per_epoch, 
                        epochs=epochs,
                        verbose=0,
                        callbacks=callbacks_list,
                        validation_data=test_data_generator,
                        validation_steps=validation_steps)
    
    with open(output_prefix + ".hist.json", 'w') as f:
        json.dump(history.history, f)
        
    
    model_file = output_prefix + ".weights.h5"
    
    model.save_weights(model_file)
    
    conv_weights = model.get_layer("shared_conv").get_weights()[0]
    
    aurocs = {}
    for i in range(n_motifs):
        motif_id = output_prefix + "_" + str(i)
        temp_conv_weights = np.zeros((w, 4, n_motifs))
        temp_conv_weights[:,:,i] = conv_weights[:,:,i]
        
        model.get_layer("shared_conv").set_weights([temp_conv_weights])
        
        eval_steps = len(test_pos_seqs) // (batch_size // 2)
        
        y_eval = np.tile(np.array([1 for i in range(batch_size//2)]+[0 for i in range(batch_size//2)]), eval_steps)
        
        
        y_pred = model.predict_generator(test_data_generator, steps=eval_steps)
        
        auc = roc_auc_score(y_eval, y_pred)
        fpr, tpr, thresholds = roc_curve(y_eval, y_pred)
        
        aurocs[motif_id] = { "auroc" : auc, "fpr" : fpr.tolist(), "tpr" : tpr.tolist() }
        
#     ppms = scan_fasta(pos_fasta, conv_weights, output_prefix, e = 0)
    ppms = scan_seqs(pos_seqs, conv_weights, output_prefix, e = 0, mode='zoops')
    
    tomtom_command = ["tomtom", "--thresh", "1.0", output_prefix + ".meme", motif_db, "--text"]
    result = run(tomtom_command, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    tomtom_results = result.stdout.split("\n")
    tomtom_results = [line.split() for line in tomtom_results if len(line.split()) > 0]

    motif_dict = {}
    for motif in ppms:
        motif_id, n, ppm = motif
        motif_dict[motif_id] = {'id' : motif_id, 
                                'n_sites' : n, "n_frac" : n / num_pos,
                                'ppm' : ppm.tolist()}
        matches = [line[1] for line in tomtom_results if line[0] == motif_id]
        if len(matches) > 0:
            motif_dict[motif_id]["matches"] = matches
            motif_dict[motif_id]["p_values"] = [float(line[3]) for line in tomtom_results if line[0] == motif_id]
        else:
            motif_dict[motif_id]["matches"] = ["No Match"]
            motif_dict[motif_id]["p_values"] = [1]

        motif_dict[motif_id]["auroc"] = aurocs[motif_id]["auroc"]
        motif_dict[motif_id]["fpr"] = aurocs[motif_id]["fpr"]
        motif_dict[motif_id]["tpr"] = aurocs[motif_id]["tpr"]
        
        
    motif_dict_sorted = OrderedDict(sorted(motif_dict.items(), key=lambda x: x[1]['auroc'], reverse=True))
    output_json = output_prefix + ".json"
    with open(output_json, "w") as f:
        f.write(json.dumps(motif_dict_sorted))
    
    

if __name__ == '__main__':
    main()
    
    
    
    