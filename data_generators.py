from sequence import encode_sequence
from numpy.random import seed
import numpy as np
import random
#from altschulEriksonDinuclShuffle import dinuclShuffle
from dinuclShuffle import dinuclShuffle
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

