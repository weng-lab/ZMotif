import numpy as np

class Sequence():
    def __init__(self, seq, label, score=None):
        self.seq = seq
        self.encoded_seq = encode_sequence(seq, N = [0.25, 0.25, 0.25, 0.25])
        self.label = int(label)
        self.score = float(score)
        self.length = len(seq)
        
    def __str__(self):
        return "Seqeunce: " + self.seq[:5] + "..." + self.seq[-5:] + "; Length: {}; Label: {}; Score: {}".format(self.length, self.label, self.score) 

def encode_sequence(seq, N = [0, 0, 0, 0]):
    d = { 'A' : [1, 0, 0, 0],
          'C' : [0, 1, 0, 0],
          'G' : [0, 0, 1, 0],
          'T' : [0, 0, 0, 1],
          'N' : N }

    return np.array([d[nuc] for nuc in list(seq)]).astype('float32')


def decode_sequence(encoded_seq):
    seq_list = encoded_seq.astype('int').tolist()
    decoded_seq_list = []
    for nuc in seq_list:
        if nuc == [1, 0, 0, 0]:
            decoded_seq_list.append('A')
        elif nuc == [0, 1, 0, 0]:
            decoded_seq_list.append('C')
        elif nuc == [0, 0, 1, 0]:
            decoded_seq_list.append('G')
        elif nuc == [0, 0, 0, 1]:
            decoded_seq_list.append('T')
        else:
            decoded_seq_list.append('N')

    return "".join(decoded_seq_list)

