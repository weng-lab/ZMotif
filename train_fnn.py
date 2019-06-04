import sys
from keras.models import load_model
from models import construct_classifier
from data_generators import data_gen_from_seqs
from pyfaidx import Fasta
import numpy as np
import random
from keras.callbacks import EarlyStopping
pre_trained_model_file = sys.argv[1]
pos_fasta = sys.argv[2]
neg_fasta = sys.argv[3]
output_prefix = sys.argv[4]

num_epochs = 100
batch_size = 32

pre_trained_model = load_model(pre_trained_model_file)
conv_weights = pre_trained_model.get_layer('conv1d_1').get_weights()[0]
num_kernels = conv_weights.shape[2]
kernel_width = conv_weights.shape[0]

model = construct_classifier(num_kernels = num_kernels,
                             kernel_width = kernel_width,
                             num_dense = 32)

model.get_layer('classifier_convolution').set_weights([conv_weights])

# print(model.get_layer('classifier_convolution').get_weights())

seq_lens = []
pos_seqs = []
for seq in Fasta(pos_fasta, as_raw=True, sequence_always_upper=True):
    pos_seqs.append(seq[:])
    seq_lens.append(len(seq))
    
neg_seqs = []
for seq in Fasta(neg_fasta, as_raw=True, sequence_always_upper=True):
    neg_seqs.append(seq[:])
    seq_lens.append(len(seq))
            
seq_lens = np.array(seq_lens)
max_seq_len = np.max(seq_lens)

num_pos = len(pos_seqs)
num_neg = len(neg_seqs)

random.seed(12)
random.shuffle(pos_seqs)
random.shuffle(neg_seqs)
    
pos_split_index = int(0.75 * num_pos)
neg_split_index = int(0.75 * num_neg)
    
pos_train_seqs = pos_seqs[:pos_split_index]
neg_train_seqs = neg_seqs[:neg_split_index]
    
pos_test_seqs = pos_seqs[pos_split_index:]
neg_test_seqs = neg_seqs[neg_split_index:]
    
train_gen = data_gen_from_seqs(pos_train_seqs,  neg_train_seqs, max_seq_len, 
                                       batch_size = batch_size, 
                                       pad_by = kernel_width, 
                                       aug_by = 0)
    
test_gen = data_gen_from_seqs(pos_test_seqs,  neg_test_seqs, max_seq_len, 
                              batch_size = batch_size, 
                              pad_by = kernel_width, 
                              aug_by = 0)  

steps_per_epoch = (0.75 * (num_pos + num_neg)) // batch_size
validation_steps = (0.25 * (num_pos + num_neg)) // batch_size
 
early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=12, restore_best_weights=False)
callbacks_list = [early_stopping]

history = model.fit_generator(train_gen,
                              steps_per_epoch=steps_per_epoch,
                              epochs=num_epochs,
                              validation_data=test_gen,
                              validation_steps=validation_steps,
                              #callbacks=callbacks_list,
                              shuffle=True,
                              use_multiprocessing=False,
                              workers=1,
                              max_queue_size=10,
                              verbose=1)

model.save(output_prefix + ".fnn.h5")
