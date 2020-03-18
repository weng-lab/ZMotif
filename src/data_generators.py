from src.sequence import encode_sequence
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
    def __init__(self, seqs, max_seq_len, seqs_per_epoch=None, batch_size = 32, augment_by=0, pad_by = 0, background = 'uniform', return_labels = True, encode_sequence=True, shuffle_seqs=True, redraw=True, weight_samples=False):
        self.seqs = seqs
        self.pos_seqs = [seq for seq in self.seqs if seq[1] == 1]
        self.neg_seqs = [seq for seq in self.seqs if seq[1] == 0]
        self.num_neg = len(self.neg_seqs)
        self.num_pos = len(self.pos_seqs)
        self.num_seqs = len(seqs)
        self.max_seq_len = max_seq_len
        
        self.seqs_per_epoch = seqs_per_epoch
        self.augment_by = augment_by
        self.batch_size = batch_size
        self.sample_size = self.batch_size // 2
        self.pad_by = pad_by
        self.n_iter = 0
        self.return_labels = return_labels
        self.encode = encode_sequence
        self.epoch = 0
        self.redraw = redraw
        self.gn_sigma = 0
        self.labels = np.array([1 for i in range(self.sample_size)] + [0 for i in range(self.sample_size)])
        if self.redraw:
            self.redraw_every = int(np.ceil(2 * len(self.neg_seqs) / self.seqs_per_epoch))
            print("Will redraw negatives every {} epochs".format(self.redraw_every))
            
        self.pos_array = 0.25 * np.ones((self.num_pos, self.max_seq_len + 2*self.pad_by, 4))
        for i, seq in enumerate(self.pos_seqs):
            if self.encode:
                seq_len = len(seq[0])
                start_index = self.pad_by + (self.max_seq_len - seq_len) // 2
                self.pos_array[i,start_index:start_index+seq_len,:] = encode_sequence(seq[0], N = [0.25, 0.25, 0.25, 0.25])
            else:
                seq_len = seq[0].shape[0]
                start_index = self.pad_by + (self.max_seq_len - seq_len) // 2
                self.pos_array[i,start_index:start_index+seq_len,:] = seq[0]
        
        self.neg_array = 0.25 * np.ones((self.num_neg, self.max_seq_len + 2*self.pad_by, 4))
        for i, seq in enumerate(self.neg_seqs):
            if self.encode:
                seq_len = len(seq[0])
                start_index = self.pad_by + (self.max_seq_len - seq_len) // 2
                self.neg_array[i,start_index:start_index+seq_len,:] = encode_sequence(seq[0], N = [0.25, 0.25, 0.25, 0.25])
            else:
                seq_len = seq[0].shape[0]
                start_index = self.pad_by + (self.max_seq_len - seq_len) // 2
                self.neg_array[i,start_index:start_index+seq_len,:] = seq[0]
                                                                                      
                                                                                      
                                                                                      
    
    def __len__(self):
        return int(np.ceil(self.seqs_per_epoch / float(self.batch_size)))
    
    def get_steps_per_epoch(self):
        return int(np.ceil(self.seqs_per_epoch / float(self.batch_size)))
    
    def set_return_labels(self, val):
        self.return_labels = val
    
    def on_epoch_end(self):
        self.epoch += 1
        np.random.shuffle(self.pos_array)
        np.random.shuffle(self.neg_array)
#         if self.redraw:
#             if (self.epoch % self.redraw_every) == 0:
#                 #print("Redrawing negatives")
#                 self.neg_seqs = []
#                 for seq in self.pos_seqs:
#                     self.neg_seqs.append([dinuclShuffle(seq[0]), 0, seq[2], seq[3], seq[4]])
    @staticmethod
    def get_sample__(big_list, num_elements, sample_size, n_iter):
        start_index = (n_iter * sample_size) % (num_elements - sample_size)
        if start_index + sample_size < (num_elements - sample_size):
            return(big_list[start_index:start_index + sample_size])
        else:
            random.shuffle(big_list)
            start_index = np.random.randint(0, num_elements - sample_size - 1)
            return(big_list[start_index:start_index + sample_size])

    def __getitem__(self, idx):
        pos_index = (self.n_iter * self.sample_size) % (self.num_pos - self.sample_size)
        neg_index = (self.n_iter * self.sample_size) % (self.num_neg - self.sample_size)
        
        pos_sample = self.pos_array[pos_index:pos_index+self.sample_size,:,:]
        neg_sample = self.neg_array[neg_index:neg_index+self.sample_size,:,:]
        output = np.vstack((pos_sample, neg_sample))
        self.n_iter += 1
        
#         pos_sample = self.get_sample__(self.pos_seqs, len(self.pos_seqs), self.sample_size, self.n_iter)
#         neg_sample = self.get_sample__(self.neg_seqs, len(self.neg_seqs), self.sample_size, self.n_iter)
#         batch_seqs = pos_sample + neg_sample
#         self.n_iter += 1
#         output = 0.25 * np.ones((self.batch_size, self.max_seq_len + 2*self.pad_by, 4))
#         labels = []
#         weights = []
#         if self.encode:
#             start_indices = np.array([((self.max_seq_len - len(seq[0])) // 2) for seq in batch_seqs])
#         else:
#             start_indices = np.array([((self.max_seq_len - seq[0].shape[0]) // 2) for seq in batch_seqs])
        
#         start_indices += self.pad_by
#         for i, seq in enumerate(batch_seqs):
#             if self.encode:
#                 seq_len = len(seq[0])
#             else:
#                 seq_len = seq[0].shape[0]
            
#             start_index = start_indices[i]
            
#             if self.encode:
#                 output[i,start_index:(start_index + seq_len),:] = encode_sequence(seq[0], N = [0.25, 0.25, 0.25, 0.25])
#             else:
#                 output[i,start_index:(start_index + seq_len),:] = seq[0]

#             if self.gn_sigma > 0:
#                 output = gaussian_filter(output, signal = self.gn_sigma)
                
#             labels.append(seq[1])
#             weights.append(seq[2])

        
        if self.return_labels:
#             return(output, np.array(labels), np.array(weights))
            return(output, self.labels)
        else:
            return(output)
