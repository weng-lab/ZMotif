from sequence import encode_sequence
from numpy.random import seed
import numpy as np
import random
#from altschulEriksonDinuclShuffle import dinuclShuffle
from dinuclShuffle import dinuclShuffle
from keras.utils import Sequence

def get_sample_from_list(big_list, num_elements, sample_size, n_iter):
    start_index = (n_iter * sample_size) % num_elements
    if start_index + sample_size < num_elements:
        return(big_list[start_index:start_index + sample_size])
    else:
        random.shuffle(big_list)
        start_index = np.random.randint(0, num_elements - sample_size - 1)
        return(big_list[start_index:start_index + sample_size])


def data_gen_from_seqs(pos_seqs, neg_seqs, max_seq_len, batch_size = 32, pad_by = 0, aug_by = 0, shuffle_negs = False):
    seed(12)
    num_pos = len(pos_seqs)
    num_neg = len(neg_seqs)
    n_iter = 0
    sample_size = batch_size // 2
    labels = np.array([1 for i in range(sample_size)] + [0 for i in range(sample_size)])
    while True:
        if aug_by > 0:
            pos_sample = get_sample_from_list(pos_seqs, num_pos, sample_size, n_iter)
            if shuffle_negs:
                if n_iter % (num_neg // sample_size) == 0:
                    print("Shuffling negative sequences")
                    neg_seqs = [dinuclShuffle(seq) for seq in pos_seqs]
            neg_sample = get_sample_from_list(neg_seqs, num_neg, sample_size, n_iter)
            n_iter += 1
            output = 0.25 * np.ones((batch_size, max_seq_len + pad_by + aug_by, 4))
            start_indices = np.random.randint(0, aug_by, size = batch_size)
            for i, seq in enumerate(pos_sample + neg_sample):
                seq_len = len(seq)
                start_index = pad_by + start_indices[i]
                output[i,start_index:(start_index + seq_len),:] = encode_sequence(seq, N = [0.25, 0.25, 0.25, 0.25])
            yield(output, labels)
        else:
            pos_sample = get_sample_from_list(pos_seqs, num_pos, sample_size, n_iter)
            neg_sample = get_sample_from_list(neg_seqs, num_neg, sample_size, n_iter)
            n_iter += 1
            output = 0.25 * np.ones((batch_size, max_seq_len + pad_by + aug_by, 4))
            for i, seq in enumerate(pos_sample + neg_sample):
                seq_len = len(seq)
                start_index = pad_by
                output[i,start_index:(start_index + seq_len),:] = encode_sequence(seq, N = [0.25, 0.25, 0.25, 0.25])
            yield(output, labels)

class DataGenerator(Sequence):
    def __init__(self, pos_seqs, neg_seqs, max_seq_len, batch_size = 32, augment_by=0, pad_by = 0, background = 'uniform'):
        self.pos_seqs = pos_seqs
        self.neg_seqs = neg_seqs
        self.num_pos = len(self.pos_seqs)
        self.num_neg = len(self.neg_seqs)
        self.max_seq_len = max_seq_len
        self.augment_by = augment_by
        self.batch_size = batch_size
        self.pad_by = pad_by
        if background == 'uniform':
            self.background = [0.25, 0.25, 0.25, 0.25]
        else:
            self.background = [0.0, 0.0, 0.0, 0.0]
        
        self.n_iter = 0
        self.sample_size = self.batch_size // 2
        self.labels = np.array([1 for i in range(self.sample_size)] + [0 for i in range(self.sample_size)])
    
    def __len__(self):
        return int(np.ceil(len(self.pos_seqs + self.neg_seqs) / float(self.batch_size)))
    
    @staticmethod
    def get_sample__(big_list, num_elements, sample_size, n_iter):
        start_index = (n_iter * sample_size) % num_elements
        if start_index + sample_size < num_elements:
            return(big_list[start_index:start_index + sample_size])
        else:
            random.shuffle(big_list)
            start_index = np.random.randint(0, num_elements - sample_size - 1)
            return(big_list[start_index:start_index + sample_size])
    
    def set_batch_size(self, new_batch_size):
        self.batch_size = new_batch_size
        self.sample_size = self.batch_size // 2
        self.labels = np.array([1 for i in range(self.sample_size)] + [0 for i in range(self.sample_size)])
    def __getitem__(self, idx):
        pos_sample = self.get_sample__(self.pos_seqs, self.num_pos, self.sample_size, self.n_iter)
        neg_sample = self.get_sample__(self.neg_seqs, self.num_neg, self.sample_size, self.n_iter)
        self.n_iter += 1
        output = 0.25 * np.ones((self.batch_size, self.max_seq_len + 2*self.pad_by + self.augment_by, 4))
        start_indices = np.random.randint(0, self.augment_by, size = self.batch_size)
        for i, seq in enumerate(pos_sample + neg_sample):
            seq_len = len(seq)
            start_index = self.pad_by + start_indices[i]
            output[i,start_index:(start_index + seq_len),:] = encode_sequence(seq, N = [0.25, 0.25, 0.25, 0.25])
        return(output, self.labels)
        
        
        
    