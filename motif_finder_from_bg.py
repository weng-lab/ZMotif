def construct_argument_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Motif Finder')
    parser.add_argument('-i', '--input', help='Bed or bedgraph containing positive regions', type=str, required=True)
    parser.add_argument('-g', '--genome', help='Genome fasta file', type=str, required=True)
    parser.add_argument('-o', '--output_prefix', help='Prefix of output files', type=str, required=True)
    parser.add_argument('-e', '--num_epochs', help='Number of epochs to train', type=int, required=False, default=2000)
    parser.add_argument('-l', '--max_seq_length', help='Trim sequences greater than', type=str, required=False, default=500)
    parser.add_argument('-int', '--intervals_per_epoch', help='Number of intervals to train on per epoch', type=int, required=False, default=5000)
    parser.add_argument('-aug', '--aug_by', help='Perform data augmentation on training sequences', type=int, required=False, default=40)
    parser.add_argument('-b', '--batch_size', help='Batch size for training', type=int, required=False, default=32)
    parser.add_argument('-split', '--train_test_split', help='Proportion of data to use for training', type=float, required=False, default=0.90)
    parser.add_argument('-k', '--num_kernels', help='Number of convolution kernels (motifs)', type=int, required=False, default=16)
    parser.add_argument('-w', '--kernel_width', help='Width of convolution kernels', type=int, required=False, default=40)
    parser.add_argument('-c', '--cycle_length', help='Cycle length for cyclical learning rate', type=int, required=False, default=1)
    parser.add_argument('-a', '--kernel_activation', help='Kernel activation function', type=str, required=False, default='linear')
    parser.add_argument('-gn', '--gaussian_noise', help='Absolute value of uniform distribution to draw noise from', type=float, required=False, default=0.0)
    parser.add_argument('-l1', '--l1_reg', help='L1 Regularization', type=float, required=False, default=0.00000001) 
    parser.add_argument('-swa', '--swa_start', help='Epoch to start stochastic weight averaging', required = False, type = int, default = None)
    parser.add_argument('-max_lr', '--max_learning_rate', help='Maximum learning rate', required = False, type = float, default = 0.1)
    parser.add_argument('-min_lr', '--min_learning_rate', help='Minimum learning rate', required = False, type = float, default = 0.01)
    parser.add_argument('-train_only', '--train_only', help='Only train classifier', required = False, type = bool, default = False)
    parser.add_argument('-mode', '--mode', help='Mode for number of motif instances per sequence', required = False, type = str, choices = ["zoops", "anr"], default = "anr")
    parser.add_argument('-curr', '--curriculum_mode', help='Use curriculum learning', required = False, type = bool, default = False)
    return parser


def main():
    from numpy.random import seed
    seed(12)
    from tensorflow import set_random_seed
    set_random_seed(12)
    
    from preprocess import bg_to_seqs, bg_to_fasta
    from data_generators import DataGeneratorBg, DataGeneratorCurriculum
    from custom_callbacks import SWA, SGDRScheduler, SequentialKernelAddition
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
    from tensorflow.contrib.opt import AdamWOptimizer
    #from altschulEriksonDinuclShuffle import dinuclShuffle
    from dinuclShuffle import dinuclShuffle
    import json
    from keras.optimizers import Adam
    from sklearn.metrics import roc_auc_score, roc_curve
    import pickle
    
    args = construct_argument_parser().parse_args()
    input_file = args.input
    genome_fasta = args.genome
    max_seq_len = int(args.max_seq_length)
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
    l1_reg = args.l1_reg
    swa_start = args.swa_start
    max_learning_rate = args.max_learning_rate
    min_learning_rate = args.min_learning_rate
    train_only = args.train_only
    mode = args.mode
    use_curriculum = True
    store_encoded = True
    
    encode_sequence = True
    if store_encoded:
        encode_sequence = False
    
    print(encode_sequence)
    seqs = bg_to_seqs(input_file, genome_fasta, max_seq_len, store_encoded=store_encoded)  
    bg_to_fasta(input_file, genome_fasta, output_prefix + ".pos.fasta")
    
    num_pos = len([seq for seq in seqs if seq[1] == 1])
    num_neg = len([seq for seq in seqs if seq[1] == 0])
    # Adjust sample weights based on relative class frequencies
    for i, seq in enumerate(seqs):
        if seq[1] == 1:
            seqs[i][2] = num_neg / num_pos
            
    
    # Adjsut sample weights based on rank
    seqs.sort(key= lambda seq: seq[3], reverse=True)
#     for i, seq in enumerate(seqs):
#         if seq[1] == 1:
#             seqs[i][2] = seqs[i][2] - (i/(num_pos+num_neg))
#             seqs[i][2] = seqs[i][2] * (1 - i/(num_pos+num_neg))
            
    random.shuffle(seqs)
    train_seqs = seqs[:int(train_test_split*len(seqs))]
    test_seqs = seqs[int(train_test_split*len(seqs)):]
    
    if intervals_per_epoch is not None:
        steps_per_epoch = intervals_per_epoch // batch_size
        validation_steps = steps_per_epoch // 3
    else:
        steps_per_epoch = len(train_seqs) // batch_size
        validation_steps = len(test_seqs) // batch_size
    
    print("Train steps: {}".format(steps_per_epoch))
    print("Test steps: {}".format(validation_steps))
    #steps_per_epoch = np.ceil((intervals_per_epoch / batch_size))
    validation_steps = int((1 - train_test_split) * steps_per_epoch)
    
    train_seqs.sort(reverse=True, key= lambda seq: seq[3]) 
    
    if not use_curriculum:
        print("I'm here")
        train_gen = DataGeneratorBg(train_seqs, max_seq_len, batch_size=batch_size,
                                  augment_by=aug_by, pad_by=kernel_width, seqs_per_epoch=intervals_per_epoch,
                                   encode_sequence=encode_sequence)

        test_gen = DataGeneratorBg(test_seqs, max_seq_len, batch_size=batch_size,
                                  augment_by=aug_by, pad_by=kernel_width, seqs_per_epoch=validation_steps*batch_size,
                                  encode_sequence=encode_sequence)
        
    else:
        train_gen = DataGeneratorCurriculum(train_seqs, max_seq_len, batch_size=batch_size,
                                  augment_by=aug_by, pad_by=kernel_width, seqs_per_epoch=intervals_per_epoch, encode_sequence=encode_sequence)

        test_gen = DataGeneratorBg(test_seqs, max_seq_len, batch_size=batch_size,
                                  augment_by=aug_by, pad_by=kernel_width, seqs_per_epoch=validation_steps*batch_size, encode_sequence=encode_sequence)
    
    
    ## Define callbacks
    
    adamw = AdamW(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0., weight_decay=0.025, batch_size=batch_size, samples_per_epoch=(batch_size * steps_per_epoch), epochs=num_epochs)
    
    model = construct_model(num_kernels=num_kernels, kernel_width=kernel_width, seq_len=None, optimizer=adamw, 
                            activation='linear', l1_reg=l1_reg, gaussian_noise = gaussian_noise)
    
    dense_weights = model.get_layer('dense_1').get_weights()[0]
    model.get_layer('dense_1').set_weights([0.01 * dense_weights])
    
    swa = SWA(num_epochs, prop = 0.2, interval = 1)
    
    
    schedule = SGDRScheduler(min_lr=min_learning_rate,
                             max_lr=max_learning_rate,
                             steps_per_epoch=steps_per_epoch,
                             lr_decay=1.0,
                             cycle_length=cycle_length,
                             mult_factor=1.0)
    
    ska = SequentialKernelAddition()
#     early_stopping = EarlyStopping(monitor='val_loss',
#                                    patience=5,
#                                    verbose=0,
#                                    restore_best_weights=True)
    
#     callbacks_list = [schedule, swa, ska]
    callbacks_list = [schedule, swa]
    
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
#     for i in range(num_kernels):
#         sample_size = batch_size // 2
#         temp_conv_weights = np.zeros((kernel_width, 4, num_kernels))
#         temp_conv_weights[:,:,i] = conv_weights[:,:,i]
        
#         model.get_layer('conv1d_1').set_weights([temp_conv_weights])
        
#         eval_gen = DataGenerator2(test_seqs, max_seq_len, batch_size=batch_size,
#                               augment_by=aug_by, pad_by=kernel_width, return_labels=False)
#         eval_steps = int(np.floor(len(test_seqs) / batch_size))
        
#         y_eval = np.array([seq[1] for seq in test_seqs[:int(batch_size*eval_steps)]])
        
        
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
        pos_hits = scan_fasta_for_kernels(output_prefix + ".pos.fasta", conv_weights, output_prefix, mode = mode, scan_pos_only = True)
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
        
        ppms_rc = []
        for ppm in raw_ppms:
            rc_ppm = ppm[2][::-1,::-1]
            ppms_rc.append((ppm[0], ppm[1], rc_ppm))
        
        rc_meme_file = output_prefix + ".rc.meme"
        ppms_to_meme(ppms_rc, rc_meme_file)
            
            
            
    
if __name__ == '__main__':
    main()
