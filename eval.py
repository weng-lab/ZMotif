import sys
import numpy as np
from keras.models import load_model
from src.sequence import encode_sequence
from sklearn.metrics import roc_auc_score
from pyfaidx import Fasta
from src.models import construct_model
import h5py

weights_file = sys.argv[1]
pos_fasta = sys.argv[2]
neg_fasta = sys.argv[3]
acc = sys.argv[4]

def read_hdf5(path):

    weights = {}

    keys = []
    with h5py.File(path, 'r') as f: # open file
        f.visit(keys.append) # append all keys to list
        for key in keys:
            if ':' in key: # contains data if ':' in key
                print(f[key].name)
                weights[f[key].name] = f[key].value
    return weights

model_weights = read_hdf5(weights_file)
conv_weights = model_weights['/conv1d_1/conv1d_1/kernel:0']
dense_weights = model_weights['/dense_1/dense_1/kernel:0']
print(conv_weights.shape)
print(dense_weights.shape)
w = conv_weights.shape[0]
k = conv_weights.shape[2]



model = construct_model(num_kernels=k, kernel_width=w)
model.load_weights(weights_file)


# new_conv_weights = np.zeros((w, 4, k))
# for index in [10,6,11,27,5,26]:
#     new_conv_weights[:,:,index] = conv_weights[:,:,index]

# model.get_layer("conv1d_1").set_weights([new_conv_weights])

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
