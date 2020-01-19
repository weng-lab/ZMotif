def construct_argument_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Motif Finder')
    parser.add_argument('-p', '--pos_fasta', help='Fasta file containing positive training sequences', type=str, required=True)
    parser.add_argument('-n', '--neg_fasta', help='Fasta file containing negative training sequences', type=str, required=False, default=None)
    parser.add_argument('-m', '--model_file', help='Model file (if provided, a new one will not be trained)', type=str, required=False, default=None)
    parser.add_argument('-o', '--output_prefix', help='Prefix of output files', type=str, required=True)
    parser.add_argument('-e', '--num_epochs', help='Number of epochs to train', type=int, required=False, default=2000)
    parser.add_argument('-int', '--intervals_per_epoch', help='Number of intervals to train on per epoch', type=int, required=False, default=5000)
    parser.add_argument('-aug', '--aug_by', help='Perform data augmentation on training sequences', type=int, required=False, default=40)
    parser.add_argument('-b', '--batch_size', help='Batch size for training', type=int, required=False, default=32)
    parser.add_argument('-split', '--train_test_split', help='Proportion of data to use for training', type=float, required=False, default=0.90)
    parser.add_argument('-k', '--num_kernels', help='Number of convolution kernels (motifs)', type=int, required=False, default=16)
    parser.add_argument('-w', '--kernel_width', help='Width of convolution kernels', type=int, required=False, default=40)
    parser.add_argument('-c', '--cycle_length', help='Cycle length for cyclical learning rate', type=int, required=False, default=1)
    parser.add_argument('-a', '--kernel_activation', help='Kernel activation function', type=str, required=False, default='linear')
    parser.add_argument('-gn', '--gaussian_noise', help='Absolute value of uniform distribution to draw noise from', type=float, required=False, default=0.0)
    parser.add_argument('-l1', '--l1_reg', help='L1 Regularization', type=float, required=False, default=0.000001) 
    parser.add_argument('-swa', '--swa_start', help='Epoch to start stochastic weight averaging', required = False, type = int, default = None)
    parser.add_argument('-max_lr', '--max_learning_rate', help='Maximum learning rate', required = False, type = float, default = 0.1)
    parser.add_argument('-min_lr', '--min_learning_rate', help='Minimum learning rate', required = False, type = float, default = 0.01)
    parser.add_argument('-train_only', '--train_only', help='Only train classifier', required = False, type = bool, default = False)
    parser.add_argument('-mode', '--mode', help='Mode for number of motif instances per sequence', required = False, type = str, choices = ["zoops", "anr"], default = "anr")
    parser.add_argument('-reinit', '--use_reinitializer', help='Reinitialize poor kernels', required = False, type = bool, default = False)
    parser.add_argument('-spd', '--spatial_dropout', help='Proportion of kernels to dropout each batch', required = False, type = float, default = 0.0)
    return parser


def main():
    from numpy.random import seed
    seed(12)
    from tensorflow import set_random_seed
    set_random_seed(12)
    from data_generators import DataGeneratorFasta, DataGeneratorDinucShuffle, DataGenerator
    from custom_callbacks import SWA, SGDRScheduler, OverfitMonitor
    from models import construct_model
    from postprocess import scan_fasta_for_kernels, hits_to_ppms
    from motif import ppms_to_meme
    from pybedtools import BedTool
    from pyfaidx import Fasta
    import numpy as np
    import random
    from keras.callbacks import EarlyStopping
    from keras import backend as K
    from AdamW import AdamW
    #from altschulEriksonDinuclShuffle import dinuclShuffle
    from dinuclShuffle import dinuclShuffle
    import json
    from keras.optimizers import Adam
    from sklearn.metrics import roc_auc_score, roc_curve
    import pickle
    
    args = construct_argument_parser().parse_args()
    pos_fasta = args.pos_fasta
    neg_fasta = args.neg_fasta
    model_file = args.model_file
    output_prefix = args.output_prefix
    num_epochs = args.num_epochs
    intervals_per_epoch = args.intervals_per_epoch
    aug_by = args.aug_by
    batch_size = args.batch_size
    train_test_split = args.train_test_split
    num_kernels = args.num_kernels
    kernel_width = args.kernel_width
    cycle_length = args.cycle_length
    kernel_activation = args.kernel_activation
    gaussian_noise = args.gaussian_noise
    print(gaussian_noise)
    l1_reg = args.l1_reg
    swa_start = args.swa_start
    max_learning_rate = args.max_learning_rate
    min_learning_rate = args.min_learning_rate
    train_only = args.train_only
    mode = args.mode
    reinit = args.use_reinitializer
    spatial_dropout = args.spatial_dropout
    
    # sequences are stored as a list of strings
    pos_seqs = []
    for seq in Fasta(pos_fasta, as_raw=True, sequence_always_upper=True):
        pos_seqs.append(seq[:])
    
    if neg_fasta is not None:
        neg_seqs = []
        for seq in Fasta(neg_fasta, as_raw=True, sequence_always_upper=True):
            neg_seqs.append(seq[:])
    
    
    # get maximum sequence length
    
    if neg_fasta is not None:
        seq_lens = np.array([len(seq) for seq in pos_seqs + neg_seqs])
    else:
        seq_lens = np.array([len(seq) for seq in pos_seqs])
        
    max_seq_len = np.max(seq_lens)
    
    num_pos = len(pos_seqs)
    
    if neg_fasta is not None:
        num_neg = len(neg_seqs)
        
    random.shuffle(pos_seqs)
    if neg_fasta is not None:
        random.shuffle(neg_seqs)
    
    pos_train_seqs = pos_seqs[:int(train_test_split*len(pos_seqs))]
    print(len(pos_train_seqs))
    pos_test_seqs = pos_seqs[int(train_test_split*len(pos_seqs)):]
    print(len(pos_test_seqs))
    if neg_fasta is not None:
        neg_train_seqs = neg_seqs[:int(train_test_split*len(neg_seqs))]
        neg_test_seqs = neg_seqs[int(train_test_split*len(neg_seqs)):]
   
    
    
    if neg_fasta is not None:
#         train_gen = DataGeneratorFasta(pos_train_seqs, max_seq_len, neg_seqs=neg_train_seqs, batch_size=batch_size,
#                                        augment_by=aug_by, pad_by=kernel_width)
    
#         test_gen = DataGeneratorFasta(pos_test_seqs, max_seq_len, neg_seqs=neg_test_seqs, batch_size=batch_size,
#                                       augment_by=aug_by, pad_by=kernel_width)

#         train_gen = data_gen_from_seqs(pos_train_seqs, neg_train_seqs, max_seq_len, batch_size=batch_size,
#                                        aug_by=aug_by, pad_by=kernel_width)
    
#         test_gen = data_gen_from_seqs(pos_test_seqs, neg_test_seqs, max_seq_len, batch_size=batch_size,
#                                       aug_by=aug_by, pad_by=kernel_width)
        
        train_gen = DataGenerator(pos_train_seqs, neg_train_seqs, max_seq_len, batch_size=batch_size,
                                       augment_by=aug_by, pad_by=kernel_width)
    
        test_gen = DataGenerator(pos_test_seqs, neg_test_seqs, max_seq_len, batch_size=batch_size,
                                      augment_by=aug_by, pad_by=kernel_width)

    else:
#         train_gen = DataGeneratorFasta(pos_train_seqs, max_seq_len, batch_size=batch_size,
#                                        augment_by=aug_by, pad_by=kernel_width)
    
#         test_gen = DataGeneratorFasta(pos_test_seqs, max_seq_len, batch_size=batch_size,
#                                       augment_by=aug_by, pad_by=kernel_width)

        train_gen = DataGeneratorDinucShuffle(pos_train_seqs, max_seq_len, batch_size=batch_size,
                                       augment_by=aug_by, pad_by=kernel_width, reshuffle=True)
    
        test_gen = DataGeneratorDinucShuffle(pos_test_seqs, max_seq_len, batch_size=batch_size,
                                      augment_by=aug_by, pad_by=kernel_width, reshuffle=False)

        
        
    
    ## Define callbacks
    if len(pos_seqs) < 2500:
        intervals_per_epoch = 2 * len(pos_seqs)
        print("Setting sequences per epoch to {}".format(intervals_per_epoch))
    steps_per_epoch = np.ceil((intervals_per_epoch / batch_size))
    validation_steps = int((1 - train_test_split) * steps_per_epoch)
    
    K.set_session(K.tf.Session(config=K.tf.ConfigProto(device_count={'CPU': 4},
                               allow_soft_placement=True,
                               intra_op_parallelism_threads=4,
                               inter_op_parallelism_threads=2)))

        
    adamw = AdamW(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0., weight_decay=0.025, batch_size=batch_size, samples_per_epoch=(batch_size * steps_per_epoch), epochs=num_epochs)
    
    model = construct_model(num_kernels=num_kernels, kernel_width=kernel_width, seq_len=None, optimizer=adamw, 
                            activation='linear', l1_reg=l1_reg, gaussian_noise = gaussian_noise, spatial_dropout=spatial_dropout)
    
    dense_weights = model.get_layer('dense_1').get_weights()[0]
    model.get_layer('dense_1').set_weights([0.01 * dense_weights])
    
    swa = SWA(num_epochs, prop = 0.2, interval = 1)
    
    
    schedule = SGDRScheduler(min_lr=min_learning_rate,
                             max_lr=max_learning_rate,
                             steps_per_epoch=steps_per_epoch,
                             lr_decay=1.0,
                             cycle_length=cycle_length,
                             mult_factor=1.0,
                             shape="cosine")
    

   
    overfit_monitor = OverfitMonitor()
    if reinit:
        reinitializer = Reinitializer(pos_test_seqs)
    #callbacks_list = [schedule, swa, early_stopping]
    #callbacks_list = [schedule, swa, overfit_monitor]
    if reinit:
        callbacks_list = [schedule, overfit_monitor, reinitializer, swa]
    else:
        callbacks_list = [schedule, overfit_monitor, swa]

    history = model.fit_generator(train_gen,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=num_epochs,
                                  validation_data=test_gen,
                                  validation_steps=validation_steps,
                                  callbacks=callbacks_list,
                                  shuffle=False,
                                  use_multiprocessing=False,
                                  workers=1,
                                  max_queue_size=10,
                                  verbose=2)
    
    model_file = output_prefix + ".weights.h5"
    model.save_weights(model_file)
    
    conv_weights = model.get_layer('conv1d_1').get_weights()[0]
    
#     kernel_aurocs = []
#     roc_curves = {}
#     # Get overall model auroc
#     if neg_fasta is not None:
#         eval_gen = DataGeneratorFasta(pos_test_seqs, max_seq_len, neg_seqs=neg_test_seqs, batch_size=batch_size,
#                                   augment_by=aug_by, pad_by=kernel_width, return_labels=False)
#         eval_steps = int(np.floor(len(pos_test_seqs + neg_test_seqs) / batch_size))
#     else:
#         eval_gen = DataGeneratorDinucShuffle(pos_test_seqs, max_seq_len, batch_size=batch_size,
#                                   augment_by=aug_by, pad_by=kernel_width, return_labels=False, reshuffle = False)
#         eval_steps = int(np.floor(2 * len(pos_test_seqs) / batch_size))

#     y_eval = np.array([1 for i in range(batch_size // 2)] + [0 for i in range(batch_size // 2)])
#     y_eval = np.tile(y_eval, eval_steps)


#     y_pred = model.predict_generator(eval_gen, steps=eval_steps)

#     fpr, tpr, t = roc_curve(y_eval, y_pred)
#     roc_curves["model"] = np.array((fpr, tpr))

#     auc = roc_auc_score(y_eval, y_pred)
#     print(auc)
#     kernel_aurocs.append(auc)

#     for i in range(num_kernels):
#         sample_size = batch_size // 2
#         temp_conv_weights = np.zeros((kernel_width, 4, num_kernels))
#         temp_conv_weights[:,:,i] = conv_weights[:,:,i]

#         model.get_layer('conv1d_1').set_weights([temp_conv_weights])

#         y_pred = model.predict_generator(eval_gen, steps=eval_steps)

#         fpr, tpr, t = roc_curve(y_eval, y_pred)
#         roc_curves[i] = np.array((fpr, tpr))

#         auc = roc_auc_score(y_eval, y_pred)
#         print(auc)
#         kernel_aurocs.append(auc)


#     kernel_aurocs = np.array(kernel_aurocs)

#     np.savetxt(output_prefix + ".kernel_aurocs", kernel_aurocs)
#     pickle.dump(roc_curves, open(output_prefix + ".roc_curves.pkl", "wb"))
    if not train_only:
        bed_file = output_prefix + ".bed"
        pos_hits = scan_fasta_for_kernels(pos_fasta, conv_weights, output_prefix, mode = mode, scan_pos_only = True)
        #neg_hits = scan_fasta_for_kernels(neg_fasta, model_file, output_prefix, scan_pos_only = True)
        with open(bed_file, "w") as f:
            #for hit in pos_hits + neg_hits:
            for hit in pos_hits:
                f.write("\t".join([str(x) for x in hit]) + "\n")



        raw_ppms, trimmed_ppms = hits_to_ppms(pos_hits)

        raw_meme_file = output_prefix + ".raw.meme"
        trimmed_meme_file = output_prefix + ".trimmed.meme"

        ppms_to_meme(raw_ppms, raw_meme_file)
        ppms_to_meme(trimmed_ppms, trimmed_meme_file)

if __name__ == '__main__':
    main()
