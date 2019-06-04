def construct_argument_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Motif Finder')
    parser.add_argument('-p', '--pos_fasta', help='Fasta file containing postive sequences', type=str, required=True)
    parser.add_argument('-o', '--output_prefix', help='Prefix of output files', type=str, required=True)
    parser.add_argument('-n', '--neg_fasta', help='Fasta file containing negative sequences', type=str, required=True)
    parser.add_argument('-e', '--num_epochs', help='Number of epochs to train', type=int, required=False, default=100)
    parser.add_argument('-aug', '--aug_by', help='Perform data augmentation on training sequences', type=int, required=False, default=0)
    parser.add_argument('-b', '--batch_size', help='Batch size for training', type=int, required=False, default=32)
    parser.add_argument('-split', '--train_test_split', help='Proportion of data to use for training', type=float, required=False, default=0.75)
    parser.add_argument('-k', '--num_kernels', help='Number of convolution kernels (motifs)', type=int, required=False, default=12)
    parser.add_argument('-w', '--kernel_width', help='Width of convolution kernels', type=int, required=False, default=40)
    parser.add_argument('-c', '--cycle_length', help='Cycle length for cyclical learning rate', type=int, required=False, default=5)
    parser.add_argument('-a', '--kernel_activation', help='Kernel activation function', type=str, required=False, default='linear')
    parser.add_argument('-d', '--drop_out', help='Dropout proportion', type=float, required=False, default=0.0)
    parser.add_argument('-eval', '--eval_fasta', help='Fasta file containing held out positive sequences', type=str, required=False, default=None)
    parser.add_argument('-data_gen', '--data_generator', help='Which data generator to use', type=int, required=False, default=1)
    parser.add_argument('-gn', '--gaussian_noise', help='Absolute value of uniform distribution to draw noise from', type=float, required=False, default=0.0)
    parser.add_argument('-l1', '--l1_reg', help='L1 Regularization', type=float, required=False, default=.0000001) 
    parser.add_argument('-swa', '--swa_start', help='Epoch to start stochastic weight averaging', required = False, type = int, default = None)
    parser.add_argument('-sd', '--spatial_dropout', help='Spatial dropout proportion', required = False, type = float, default = 0.0)
    return parser


def main():
    from utils import construct_model, one_hot_encode_fasta, SGDRScheduler, conv_weights_to_meme, get_motif_matches, encode_sequence, get_consensus_seq, seq_list_to_ppm, ppms_to_meme, data_gen, SWA_new, fimo, fasta_to_array, data_gen_2, fasta_to_array_2
    from process import motif_scan
    from custom_callbacks import Reinitializer
    from random import shuffle
    import numpy as np
    from keras.callbacks import CSVLogger
    import tensorflow as tf
    from keras.backend import tensorflow_backend as K
    from sklearn.metrics import roc_auc_score, roc_curve
    from keras.initializers import RandomUniform
    import pickle
    parser = construct_argument_parser()
    args = parser.parse_args()
    pos_fasta = args.pos_fasta
    neg_fasta = args.neg_fasta
    num_epochs = args.num_epochs
    aug_by = args.aug_by
    swa_start = args.swa_start

    print(aug_by)
    batch_size = args.batch_size
    assert batch_size % 2 == 0
    train_test_split = args.train_test_split
    num_kernels = args.num_kernels
    kernel_width = args.kernel_width
    output_prefix = args.output_prefix
    cycle_length = args.cycle_length
    assert num_epochs >= cycle_length
    kernel_activation = args.kernel_activation
    drop_out = args.drop_out
    data_generator = args.data_generator
    gaussian_noise = args.gaussian_noise
    l1_reg = args.l1_reg
    spatial_dropout = args.spatial_dropout
    
    if data_generator == 1:
        pos_seq_array = fasta_to_array(pos_fasta, pad_by = kernel_width)
        neg_seq_array = fasta_to_array(neg_fasta, pad_by = kernel_width)
        num_pos = pos_seq_array.shape[0]
        num_neg = neg_seq_array.shape[0]
        np.random.shuffle(pos_seq_array)
        np.random.shuffle(neg_seq_array)
        max_seq_len = pos_seq_array.shape[1]
        pos_split_idx = int(num_pos * train_test_split)
        neg_split_idx = int(num_neg * train_test_split)
        train_gen = data_gen(pos_seq_array[:pos_split_idx,:,:], neg_seq_array[:neg_split_idx,:,:], max_seq_len, batch_size=batch_size, augment_by = aug_by)
        test_gen = data_gen(pos_seq_array[pos_split_idx:,:,:], neg_seq_array[neg_split_idx:,:,:], max_seq_len, batch_size=batch_size, augment_by = aug_by)
        class_weight = {0 : 1, 1 : num_neg / num_pos}
    else:
        #pos_seq_array = fasta_to_array(pos_fasta, pad_by = kernel_width)
        #neg_seq_array = fasta_to_array(neg_fasta, pad_by = kernel_width)
        pos_seq_array = fasta_to_array_2(pos_fasta, aug_by = aug_by, kernel_width = kernel_width)
        neg_seq_array = fasta_to_array_2(neg_fasta, aug_by = aug_by, kernel_width = kernel_width)
        seq_array = np.vstack((pos_seq_array, neg_seq_array))
        num_pos = pos_seq_array.shape[0]
        num_neg = neg_seq_array.shape[0]
        num_seqs = num_pos + num_neg
        labels = np.array([1 for i in range(num_pos)] + [0 for i in range(num_neg)])
        
        rng_state = np.random.get_state()
        np.random.shuffle(seq_array)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)

        split_idx = int(num_seqs * train_test_split)
        train_gen = data_gen_2(seq_array[:split_idx,:,:], labels[:split_idx], batch_size = batch_size, aug_by = aug_by)
        test_gen = data_gen_2(seq_array[split_idx:,:,:], labels[split_idx:], batch_size = batch_size, aug_by = aug_by)
        class_weight = {0 : 1, 1 : 1}
        
        
        
    
    
    

    


    num_seqs = num_pos + num_neg
    
    if num_pos > 2500:
        print("There were more than 5,000 positive sequences")
        steps_per_epoch = np.ceil((5000 / batch_size))
        validation_steps = np.ceil((2000 / batch_size))
    else:

        steps_per_epoch = np.ceil((num_seqs * train_test_split / batch_size))
        validation_steps = np.ceil((num_seqs * (1 - train_test_split) / batch_size))
    
    schedule = SGDRScheduler(min_lr=1e-5,
                             max_lr=1e-1,
                             steps_per_epoch=steps_per_epoch,
                             lr_decay=1.0,
                             cycle_length=cycle_length,
                             mult_factor=1.0)
    
    csv_logger = CSVLogger(output_prefix + ".training.log")
    if swa_start is None:
        print("SWA start epoch not specified...setting it to num_epochs divided by 2")
        swa_start = num_epochs // 2
    else:
        print("Starting SWA at epoch {}".format(swa_start))
        
    swa_new = SWA_new(start_epoch=swa_start, interval=cycle_length)
    #reinitializer = Reinitializer(pos_seq_array)
    #callbacks_list = [schedule, csv_logger, swa_new, reinitializer]
    callbacks_list = [schedule, csv_logger, swa_new]
    # construct model
    model = construct_model(num_kernels=num_kernels,
                            kernel_width=kernel_width,
                            seq_len=None,
                            num_dense=0,
                            dropout_prop=drop_out,
                            #kernel_initializer='glorot_uniform',
                            kernel_initializer=RandomUniform(-0.001,0.001),
                            optimizer='adam',
                            activation=kernel_activation,
                            gaussian_noise = gaussian_noise,
                            l1_reg = l1_reg,
                            spatial_dropout = spatial_dropout)
    
    #conv_weights_init = model.get_layer('conv1d_1').get_weights()[0]
    #print(np.max(conv_weights_init))
    #print(np.min(conv_weights_init))
    #quit()
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=num_epochs,
                                  validation_data=test_gen,
                                  #validation_steps=100,
                                  validation_steps=validation_steps,
                                  callbacks=callbacks_list,
                                  class_weight=class_weight,
                                  shuffle=True,
                                  use_multiprocessing=False,
                                  workers=1,
                                  max_queue_size=4,
                                  verbose=2)
    model_file = output_prefix + ".model.h5"
    model.save(model_file)
    min_num_seqs = int(0.01 * num_pos)

    
    del pos_seq_array
    del neg_seq_array
    
    #motif_scan(model_file, [pos_fasta], output_prefix, min_num_seqs=min_num_seqs)
    motif_scan(model_file, [pos_fasta, neg_fasta], output_prefix, min_num_seqs=min_num_seqs)
if __name__ == '__main__':
    from numpy.random import seed
    seed(12)
    from tensorflow import set_random_seed
    set_random_seed(12)
    main()


    
