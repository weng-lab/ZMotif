from src.sequence import encode_sequence, decode_sequence
from numpy.random import seed
import numpy as np
import random
#from altschulEriksonDinuclShuffle import dinuclShuffle
from src.dinuclShuffle import dinuclShuffle
from keras.utils import Sequence
from collections import defaultdict, Counter
import random
from scipy.ndimage import gaussian_filter

class DataGeneratorBg(Sequence):
    def __init__(self, seqs, max_seq_len, seqs_per_epoch=None, batch_size = 32, augment_by=0, pad_by = 0, background = 'uniform', return_labels = True, encode=True, shuffle_seqs=True, redraw=True, weight_samples=False):
        self.seqs = seqs
        self.pos_seqs = [seq for seq in self.seqs if seq[1] == 1]
        self.neg_seqs = [seq for seq in self.seqs if seq[1] == 0]
        
        self.num_neg = len(self.neg_seqs)
        self.num_pos = len(self.pos_seqs)
        self.num_seqs = len(seqs)
        self.max_seq_len = max_seq_len
        
        self.seqs_per_epoch = seqs_per_epoch
        
        self.batch_size = batch_size
        self.sample_size = self.batch_size // 2
        
        self.shuffle_pos_every = int(2 * np.ceil(self.num_pos) / self.seqs_per_epoch)
        if self.shuffle_pos_every < 1:
            self.shuffle_pos_every = 1
            
        self.shuffle_neg_every = int(2 * np.ceil(self.num_neg) / self.seqs_per_epoch)
        if self.shuffle_neg_every < 1:
            self.shuffle_neg_every = 1
        
        self.augment_by = augment_by
        self.pad_by = pad_by
        self.n_iter = 0
        self.return_labels = return_labels
        self.encode = encode
        self.epoch = 0
        self.redraw = redraw
        self.gn_sigma = 0
        self.labels = np.array([1 for i in range(self.sample_size)] + [0 for i in range(self.sample_size)])
        print("This is a test!")
    def __len__(self):
        return int(np.ceil(self.seqs_per_epoch / float(self.batch_size)))
    
        
    def on_train_begin(self, logs=None):
        pass
#         self.gn_sigma = 0.25
        
    def on_epoch_end(self, logs=None):
        self.epoch += 1
        if (self.epoch % self.shuffle_pos_every) == 0:
            random.shuffle(self.pos_seqs)
            
        if (self.epoch % self.shuffle_neg_every) == 0:
            random.shuffle(self.neg_seqs)
        
#         self.gn_sigma = (0.25 - (0.25 * self.epoch / 200))
#         if self.gn_sigma <= 0.0:
#             self.gn_sigma = 0.0
    
    def __getitem__(self, idx):
        pos_index = (self.n_iter * self.sample_size) % (self.num_pos - self.sample_size)
        neg_index = (self.n_iter * self.sample_size) % (self.num_neg - self.sample_size)
        
        pos_sample = self.pos_seqs[pos_index:pos_index+self.sample_size]
        neg_sample = self.neg_seqs[neg_index:neg_index+self.sample_size]
        
        batch_seqs = pos_sample + neg_sample
        self.n_iter += 1
        output = 0.25 * np.ones((self.batch_size, self.max_seq_len + 2*self.pad_by, 4))
        labels = []
        weights = []
        if self.encode:
            start_indices = np.array([((self.max_seq_len - len(seq[0])) // 2) for seq in batch_seqs])
        else:
            start_indices = np.array([((self.max_seq_len - seq[0].shape[0]) // 2) for seq in batch_seqs])
        
        start_indices += self.pad_by
        for i, seq in enumerate(batch_seqs):
            if self.encode:
                seq_len = len(seq[0])
            else:
                seq_len = seq[0].shape[0]
            
            start_index = start_indices[i]
            
            if self.encode:
                output[i,start_index:(start_index + seq_len),:] = encode_sequence(seq[0], N = [0.25, 0.25, 0.25, 0.25])
            else:
                output[i,start_index:(start_index + seq_len),:] = seq[0]
            
            if self.gn_sigma > 0.0:
                gaussian_filter(output, self.gn_sigma)
                
            labels.append(seq[1])
            weights.append(seq[2])

        if self.return_labels:
            return(output, np.array(labels), np.array(weights))
        else:
            return(output)

class DataGeneratorCurriculum(Sequence):
    def __init__(self, seqs, max_seq_len, curr_length=100, curr_steps=10, seqs_per_epoch=None, batch_size = 32, augment_by=0, pad_by = 0, background = 'uniform', return_labels = True, encode=True, shuffle_seqs=True, redraw=True, weight_samples=False):
        self.change_curr_every = curr_length / curr_steps
        self.curr_length = curr_length
        self.seqs = seqs
        self.pos_seqs = [seq for seq in self.seqs if seq[1] == 1]
        self.neg_seqs = [seq for seq in self.seqs if seq[1] == 0]
        
        self.num_neg = len(self.neg_seqs)
        self.num_pos = len(self.pos_seqs)
        self.num_seqs = len(seqs)
        self.max_seq_len = max_seq_len
        
        self.seqs_per_epoch = seqs_per_epoch
        
        self.batch_size = batch_size
        self.sample_size = self.batch_size // 2
        
        self.shuffle_pos_every = int(2 * np.ceil(self.num_pos) / self.seqs_per_epoch)
        if self.shuffle_pos_every < 1:
            self.shuffle_pos_every = 1
            
        self.shuffle_neg_every = int(2 * np.ceil(self.num_neg) / self.seqs_per_epoch)
        if self.shuffle_neg_every < 1:
            self.shuffle_neg_every = 1
        
        self.augment_by = augment_by
        self.pad_by = pad_by
        self.n_iter = 0
        self.return_labels = return_labels
        self.encode = encode
        self.epoch = 0
        self.redraw = redraw
        self.gn_sigma = 0
        self.labels = np.array([1 for i in range(self.sample_size)] + [0 for i in range(self.sample_size)])
    def __len__(self):
        return int(np.ceil(self.seqs_per_epoch / float(self.batch_size)))
    
    def on_epoch_begin(self, logs=None):
        if self.epoch == 0:
            self.pos_index = int(self.num_pos * 0.1)
            self.neg_index = int(self.num_neg * 0.1)
            self.seqs.sort(reverse=True, key= lambda seq: seq[3])
            self.pos_seqs = [seq for seq in self.seqs if seq[1] == 1][:self.pos_index]
            self.neg_seqs = [seq for seq in self.seqs if seq[1] == 0][:self.neg_index]
            self.num_neg = len(self.neg_seqs)
            self.num_pos = len(self.pos_seqs)
            self.shuffle_pos_every = int(2 * np.ceil(self.num_pos) / self.seqs_per_epoch)
            if self.shuffle_pos_every < 1:
                self.shuffle_pos_every = 1

            self.shuffle_neg_every = int(2 * np.ceil(self.num_neg) / self.seqs_per_epoch)
            if self.shuffle_neg_every < 1:
                self.shuffle_neg_every = 1
                
    def on_epoch_end(self, logs=None):
        self.epoch += 1
        if (self.epoch % self.shuffle_pos_every) == 0:
            random.shuffle(self.pos_seqs)
            
        if (self.epoch % self.shuffle_neg_every) == 0:
            random.shuffle(self.neg_seqs)
        
    
    def __getitem__(self, idx):
        pos_index = (self.n_iter * self.sample_size) % (self.num_pos - self.sample_size)
        neg_index = (self.n_iter * self.sample_size) % (self.num_neg - self.sample_size)
        
        pos_sample = self.pos_seqs[pos_index:pos_index+self.sample_size]
        neg_sample = self.neg_seqs[neg_index:neg_index+self.sample_size]
        
        batch_seqs = pos_sample + neg_sample
        self.n_iter += 1
        output = 0.25 * np.ones((self.batch_size, self.max_seq_len + 2*self.pad_by, 4))
        labels = []
        weights = []
        if self.encode:
            start_indices = np.array([((self.max_seq_len - len(seq[0])) // 2) for seq in batch_seqs])
        else:
            start_indices = np.array([((self.max_seq_len - seq[0].shape[0]) // 2) for seq in batch_seqs])
        
        start_indices += self.pad_by
        for i, seq in enumerate(batch_seqs):
            if self.encode:
                seq_len = len(seq[0])
            else:
                seq_len = seq[0].shape[0]
            
            start_index = start_indices[i]
            
            if self.encode:
                output[i,start_index:(start_index + seq_len),:] = encode_sequence(seq[0], N = [0.25, 0.25, 0.25, 0.25])
            else:
                output[i,start_index:(start_index + seq_len),:] = seq[0]

            if self.gn_sigma > 0:
                output = gaussian_filter(output, signal = self.gn_sigma)
                
            labels.append(seq[1])
            weights.append(seq[2])

        
        if self.return_labels:
            return(output, np.array(labels), np.array(weights))
        else:
            return(output)