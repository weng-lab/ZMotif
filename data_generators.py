from sequence import encode_sequence
from numpy.random import seed
import numpy as np
import random
#from altschulEriksonDinuclShuffle import dinuclShuffle
from dinuclShuffle import dinuclShuffle
from keras.utils import Sequence
from collections import defaultdict, Counter
import random

class DataGeneratorDinucShuffle(Sequence):
    def __init__(self, pos_seqs, max_seq_len, seqs_per_epoch=5000, batch_size = 32, augment_by=0, pad_by = 0, background = 'uniform', return_labels = True, reshuffle=True):
        self.pos_seqs = pos_seqs
        self.num_pos = len(self.pos_seqs)
        self.neg_seqs = [dinuclShuffle(seq) for seq in self.pos_seqs]
        self.num_neg = len(self.neg_seqs)
        self.max_seq_len = max_seq_len
        self.seqs_per_epoch = seqs_per_epoch
        self.augment_by = augment_by
        self.batch_size = batch_size
        self.pad_by = pad_by
        self.epoch = 0
        self.shuffle_every = int(np.ceil((self.num_pos + self.num_neg) / self.seqs_per_epoch))
        self.reshuffle = reshuffle
        #if self.reshuffle:
            #print("Will dinucleotide shuffle negative sequences every {} epochs".format(self.shuffle_every))
        if background == 'uniform':
            self.background = [0.25, 0.25, 0.25, 0.25]
        else:
            self.background = [0.0, 0.0, 0.0, 0.0]
        
        self.n_iter = 0
        self.sample_size = self.batch_size // 2
        self.labels = np.array([1 for i in range(self.sample_size)] + [0 for i in range(self.sample_size)])
        self.return_labels = return_labels
    def on_epoch_end(self):
        self.epoch += 1
        if self.reshuffle:
            if self.epoch % self.shuffle_every == 0:
                self.neg_seqs = [dinuclShuffle(seq) for seq in self.pos_seqs]
    
    def __len__(self):
        return int(np.ceil(self.seqs_per_epoch / float(self.batch_size)))
    
    @staticmethod
    def get_sample__(big_list, num_elements, sample_size, n_iter):
        start_index = (n_iter * sample_size) % (num_elements-sample_size)
        return(big_list[start_index:start_index + sample_size])
                
    def __getitem__(self, idx):
        pos_sample = self.get_sample__(self.pos_seqs, self.num_pos, self.sample_size, self.n_iter)
        neg_sample = self.get_sample__(self.neg_seqs, self.num_neg, self.sample_size, self.n_iter)
        self.n_iter += 1
        output = 0.25 * np.ones((self.batch_size, self.max_seq_len + 2*self.pad_by + self.augment_by, 4))
        if self.augment_by > 0:
            start_indices = np.random.randint(0, self.augment_by + 1, size = self.batch_size)
        else:
            start_indices = np.array([((self.max_seq_len - len(x)) // 2) for x in pos_sample + neg_sample])
        start_indices += self.pad_by
        for i, seq in enumerate(pos_sample + neg_sample):
            seq_len = len(seq)
            start_index = start_indices[i]
            output[i,start_index:(start_index + seq_len),:] = encode_sequence(seq, N = [0.25, 0.25, 0.25, 0.25])
        if self.return_labels:
            return(output, self.labels)
        else:
            return(output)
        
class DataGeneratorFasta(Sequence):
    def __init__(self, pos_seqs, max_seq_len, neg_seqs=None, k=1, seqs_per_epoch=5000, batch_size = 32, augment_by=0, pad_by = 0, background = 'uniform', return_labels = True):
        self.pos_seqs = pos_seqs
        if neg_seqs is not None:
            self.neg_seqs = neg_seqs
            self.markov = False
        else:
            self.k = k
            #print("Constructing Markov model")
            self.construct_markov_model()
            #print("Generating negative sequences from markov model")
            self.neg_seqs = []
            for seq in self.pos_seqs:
                self.neg_seqs.append(self.get_seq_from_markov(len(seq)))
                self.markov = True
            print(self.markov_model)
        self.num_pos = len(self.pos_seqs)
        self.num_neg = len(self.neg_seqs)
        self.max_seq_len = max_seq_len
        self.seqs_per_epoch = seqs_per_epoch
        self.augment_by = augment_by
        self.batch_size = batch_size
        self.pad_by = pad_by
        self.epoch = 0
        self.shuffle_every = int(np.ceil((self.num_pos + self.num_neg) / self.seqs_per_epoch))
        print("Will shuffle input sequences every {} epochs".format(self.shuffle_every))
        if background == 'uniform':
            self.background = [0.25, 0.25, 0.25, 0.25]
        else:
            self.background = [0.0, 0.0, 0.0, 0.0]
        
        self.n_iter = 0
        self.sample_size = self.batch_size // 2
        self.labels = np.array([1 for i in range(self.sample_size)] + [0 for i in range(self.sample_size)])
        self.return_labels = return_labels
    
    def construct_markov_model(self):
        # https://eli.thegreenplace.net/2018/elegant-python-code-for-a-markov-chain-text-generator/
        markov_model = defaultdict(Counter)

        for seq in self.pos_seqs:
            for i in range(len(seq) - self.k):
                state = seq[i:i + self.k]
                next = seq[i + self.k]
                markov_model[state][next] += 1
        
        
        #print(markov_model)
        for kmer in list(markov_model.keys()):
            if 'N' in kmer:
                del markov_model[kmer]
                #print("Deleted {} from markov model".format(kmer))
        
        for kmer in list(markov_model.keys()):
            if "N" in list(markov_model[kmer].keys()):
                del markov_model[kmer]["N"]
                #print("Deleted {}: N from markov model".format(kmer))
        
        self.markov_model = markov_model
    def get_seq_from_markov(self, seq_len):
        state = random.choice(list(self.markov_model))
        out = list(state)
        for i in range(seq_len):
            out.extend(random.choices(list(self.markov_model[state]), self.markov_model[state].values()))
            state = state[1:] + out[-1]
        return(''.join(out))
    
    def __len__(self):
        return int(np.ceil(self.seqs_per_epoch / float(self.batch_size)))
    
    def on_epoch_end(self):
        self.epoch += 1
        if self.epoch % self.shuffle_every == 0:
            print("Shuffling sequences")
            random.shuffle(self.pos_seqs)
            if self.markov:
                self.neg_seqs = []
                for seq in self.pos_seqs:
                    self.neg_seqs.append(self.get_seq_from_markov(len(seq)))
            else:
                random.shuffle(self.neg_seqs)
    
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
        if self.augment_by > 0:
            start_indices = np.random.randint(self.pad_by, self.pad_by + self.augment_by + 1, size = self.batch_size)
        else:
            start_indices = np.array([((self.max_seq_len - len(x)) // 2) for x in pos_sample + neg_sample])
            start_indices += self.pad_by
        for i, seq in enumerate(pos_sample + neg_sample):
            seq_len = len(seq)
            start_index = start_indices[i]
            output[i,start_index:(start_index + seq_len),:] = encode_sequence(seq, N = [0.25, 0.25, 0.25, 0.25])
        if self.return_labels:
            return(output, self.labels)
        else:
            return(output)

class DataGenerator(Sequence):
    def __init__(self, seq_list, max_seq_len, 
                 batch_size = 32, 
                 augment_by=0, 
                 pad_by = 0, 
                 seqs_per_epoch = -1):
        
        self.seq_list = seq_list
        self.num_seqs = len(self.seq_list)
        self.max_seq_len = max_seq_len
        self.augment_by = augment_by
        self.seqs_per_epoch = seqs_per_epoch
        self.batch_size = batch_size
        self.pad_by = pad_by      
        self.n_iter = 0 
        self.shuffle_every = len(self.seq_list) // self.batch_size
    def __len__(self):
        if self.seqs_per_epoch < 0:
            return int(np.ceil(len(self.seq_list) / self.batch_size))
        else:
            return int(np.ceil(self.seqs_per_epoch / self.batch_size))

#     def on_epoch_end(self):
#         random.shuffle(self.seq_list)
        
    def __getitem__(self, idx):
        sample_index = self.batch_size*self.n_iter % (self.num_seqs - self.batch_size)
        sample = self.seq_list[sample_index:sample_index+self.batch_size]
        self.n_iter += 1
        if self.n_iter % self.shuffle_every == 0:
            print("shuffling sequences")
            random.shuffle(self.seq_list)
        output = 0.25 * np.ones((self.batch_size, self.max_seq_len + 2*self.pad_by + self.augment_by, 4))
        labels = []
        for i, seq in enumerate(sample):
            start_index = self.pad_by + (self.max_seq_len - seq.length) // 2
            output[i,start_index:start_index+seq.length,:] = seq.encoded_seq
            labels.append(seq.label)
        return(output, np.array(labels))
    
class DataGeneratorBg(Sequence):
    def __init__(self, seqs, max_seq_len, seqs_per_epoch=None, batch_size = 32, augment_by=0, pad_by = 0, background = 'uniform', return_labels = True, encode_sequence=True, shuffle_seqs=True, redraw=True):
        self.seqs = seqs
        self.pos_seqs = [seq for seq in self.seqs if seq[1] == 1]
        self.neg_seqs = [seq for seq in self.seqs if seq[1] == 0]
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
        if self.redraw:
            self.redraw_every = int(np.ceil(2 * len(self.neg_seqs) / self.seqs_per_epoch))
            print("Will redraw negatives every {} epochs".format(self.redraw_every))
    def __len__(self):
        return int(np.ceil(self.seqs_per_epoch / float(self.batch_size)))
    
    
    def on_epoch_end(self):
        self.epoch += 1
        if self.redraw:
            if (self.epoch % self.redraw_every) == 0:
                #print("Redrawing negatives")
                self.neg_seqs = []
                for seq in self.pos_seqs:
                    self.neg_seqs.append([dinuclShuffle(seq[0]), 0, seq[2], seq[3], seq[4]])
    @staticmethod
    def get_sample__(big_list, num_elements, sample_size, n_iter):
        start_index = (n_iter * sample_size) % (num_elements - sample_size)
        if start_index + sample_size < (num_elements - sample_size):
            return(big_list[start_index:start_index + sample_size])
        else:
            random.shuffle(big_list)
            start_index = np.random.randint(0, num_elements - sample_size - 1)
            return(big_list[start_index:start_index + sample_size])

#     def on_epoch_end(self):
#         print("Sorting sequences")
#         self.seqs.sort(reverse=True, key= lambda seq: seq[3])
#         self.n_iter = 0
    def __getitem__(self, idx):
#         sample = self.get_sample__(self.seqs, len(self.seqs), self.batch_size, self.n_iter)
        pos_sample = self.get_sample__(self.pos_seqs, len(self.pos_seqs), self.sample_size, self.n_iter)
        neg_sample = self.get_sample__(self.neg_seqs, len(self.neg_seqs), self.sample_size, self.n_iter)
        self.n_iter += 1
        output = 0.25 * np.ones((self.batch_size, self.max_seq_len + 2*self.pad_by + self.augment_by, 4))
        labels = []
        weights = []
        if self.augment_by > 0:
            start_indices = np.random.randint(self.pad_by, self.pad_by + self.augment_by + 1, size = self.batch_size)
        else:
            if self.encode:
                start_indices = np.array([((self.max_seq_len - len(seq[0])) // 2) for seq in sample])
            else:
                start_indices = np.array([((self.max_seq_len - seq[0].shape[0]) // 2) for seq in sample])
            start_indices += self.pad_by
        for i, seq in enumerate(pos_sample + neg_sample):
            if self.encode:
                seq_len = len(seq[0])
            else:
                seq_len = seq[0].shape[0]
            start_index = start_indices[i]
            
            if self.encode:
                output[i,start_index:(start_index + seq_len),:] = encode_sequence(seq[0], N = [0.25, 0.25, 0.25, 0.25])
            else:
                output[i,start_index:(start_index + seq_len),:] = seq[0]
            
            labels.append(seq[1])
            weights.append(seq[2])
        if self.return_labels:
            return(output, np.array(labels), np.array(weights))
        else:
            return(output)