from models import construct_model
from pyfaidx import Fasta
import numpy as np
from sequence import encode_sequence
from sklearn.metrics import roc_auc_score

num_kernels = 32
w = 32

pos_fasta = "wgEncodeAwgTfbsHaibHepg2Hnf4asc8987V0416101UniPk.test.pos.fasta"
neg_fasta = "wgEncodeAwgTfbsHaibHepg2Hnf4asc8987V0416101UniPk.test.neg.fasta"
model = construct_model(num_kernels=num_kernels, 
                            kernel_width=w, 
                            seq_len=None, 
                            optimizer='adam', 
                            activation='linear')


model.load_weights('./results/wgEncodeAwgTfbsHaibHepg2Hnf4asc8987V0416101UniPk.weights.h5')
dense_weights = model.get_layer('dense_1').get_weights()[0]
seq_lens = []
num_pos = 0
for seq in Fasta(pos_fasta, as_raw = True, sequence_always_upper = True):
    num_pos += 1
    seq_lens.append(len(seq))

num_neg = 0
for seq in Fasta(neg_fasta, as_raw = True, sequence_always_upper = True):
    num_neg += 1
    seq_lens.append(len(seq))
    
seq_lens = np.array(seq_lens)
max_seq_len = np.max(seq_lens)

seq_array = 0.25*np.ones((num_pos + num_neg, max_seq_len + 2 * w, 4))
index = 0
for seq in Fasta(pos_fasta, as_raw = True, sequence_always_upper = True):
    seq_array[index, w:w + seq_lens[index], :] = encode_sequence(seq[:], N = [0.25, 0.25, 0.25, 0.25])
    index += 1

for seq in Fasta(neg_fasta, as_raw = True, sequence_always_upper = True):
    seq_array[index, w:w+seq_lens[index], :] = encode_sequence(seq[:], N = [0.25, 0.25, 0.25, 0.25])
    index += 1


y_true = np.array([1 for i in range(num_pos)] + [0 for i in range(num_neg)])
for i in range(num_kernels):
    temp_dense_weights = np.zeros((num_kernels,1))
    temp_dense_weights[i,0] = dense_weights[i,0]
    model.get_layer('dense_1').set_weights([temp_dense_weights])
    
    y_pred = model.predict(seq_array)

    auc = roc_auc_score(y_true, y_pred)
    print(i+1, auc)
    