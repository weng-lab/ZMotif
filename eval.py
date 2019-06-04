import sys
import numpy as np
from keras.models import load_model
from sequence import encode_sequence
from sklearn.metrics import roc_auc_score
from pyfaidx import Fasta
from models import construct_model

weights_file = sys.argv[1]
pos_fasta = sys.argv[2]
neg_fasta = sys.argv[3]
acc = sys.argv[4]

model = construct_model(num_kernels=32, kernel_width=32)
model.load_weights(weights_file)

w = 32

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
y_pred = model.predict(seq_array)

auc = roc_auc_score(y_true, y_pred)
print(auc)

with open(acc + ".auroc", "w") as f:
    f.write(str(auc) + "\n")
