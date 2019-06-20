import sys
from models import construct_model
import numpy as np
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))


model_weights_h5 = sys.argv[1]
model = construct_model(num_kernels=32, kernel_width=40)
model.load_weights(model_weights_h5)

conv_weights = model.get_layer('conv1d_1').get_weights()[0]
dense_weights = model.get_layer('dense_1').get_weights()[0]
w = conv_weights.shape[0]
num_kernels = conv_weights.shape[2]

test_kernel = conv_weights[:,:,8]
test_dense_weight = dense_weights[8]
consensus_seq = ""
for i in range(w):
    argmax = np.argmax(test_kernel[i,:])
    if argmax == 0:
        consensus_seq = consensus_seq + "A"
    elif argmax == 1:
        consensus_seq = consensus_seq + "C"
    elif argmax == 2:
        consensus_seq = consensus_seq + "G"
    else:
        consensus_seq = consensus_seq + "T"

max_kernel_value = np.sum(np.max(test_kernel, axis=1))
p_matrix = np.zeros((4,w))
for i in range(w):
    for j in range(4):
        p_matrix[j,i] = sigmoid((max_kernel_value - np.max(test_kernel[i,:]) + test_kernel[i,j])*test_dense_weight)
        
print(p_matrix)