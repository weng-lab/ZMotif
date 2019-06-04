def construct_argument_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Motif Finder')
    parser.add_argument('-bg', '--bedgraph', help='Bedgraph file containing genomic coordinates and labels', type=str, required=True)
    parser.add_argument('-g', '--genome_fasta', help='Genome fasta file', type=str, required=True)
    parser.add_argument('-m', '--model_file', help='Model file (if provided, a new one will not be trained)', type=str, required=False, default=None)
    parser.add_argument('-o', '--output_prefix', help='Prefix of output files', type=str, required=True)
    parser.add_argument('-e', '--num_epochs', help='Number of epochs to train', type=int, required=False, default=2000)
    parser.add_argument('-int', '--intervals_per_epoch', help='Number of intervals to train on per epoch', type=int, required=False, default=5000)
    parser.add_argument('-aug', '--aug_by', help='Perform data augmentation on training sequences', type=int, required=False, default=0)
    parser.add_argument('-b', '--batch_size', help='Batch size for training', type=int, required=False, default=32)
    parser.add_argument('-split', '--train_test_split', help='Proportion of data to use for training', type=float, required=False, default=0.75)
    parser.add_argument('-k', '--num_kernels', help='Number of convolution kernels (motifs)', type=int, required=False, default=16)
    parser.add_argument('-w', '--kernel_width', help='Width of convolution kernels', type=int, required=False, default=40)
    parser.add_argument('-c', '--cycle_length', help='Cycle length for cyclical learning rate', type=int, required=False, default=5)
    parser.add_argument('-a', '--kernel_activation', help='Kernel activation function', type=str, required=False, default='linear')
    parser.add_argument('-gn', '--gaussian_noise', help='Absolute value of uniform distribution to draw noise from', type=float, required=False, default=0.0)
    parser.add_argument('-l1', '--l1_reg', help='L1 Regularization', type=float, required=False, default=.000001) 
    parser.add_argument('-swa', '--swa_start', help='Epoch to start stochastic weight averaging', required = False, type = int, default = None)
    return parser


def main():
    from preprocess import bedtool_to_intervals, seq_lens_from_intervals
    from data_generators import data_gen_from_intervals
    from custom_callbacks import SWA, SGDRScheduler
    from models import construct_model
    from postprocess import scan_intervals_for_kernels, hits_to_ppms
    from motif import ppms_to_meme
    from pybedtools import BedTool
    from pyfaidx import Fasta
    import numpy as np
    from random import shuffle
    
    args = construct_argument_parser().parse_args()
    bedgraph = args.bedgraph
    genome_fasta = args.genome_fasta
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
    l1_reg = args.l1_reg
    swa_start = args.swa_start
    
    ## Convert genome to pyfaidx Fasta
    genome = Fasta(genome_fasta,
                   as_raw=True,
                   sequence_always_upper=True)

    ## Convert bedgraph to BedTool object
    bt = BedTool(bedgraph)
    
    ## Convert BedTool object to a list of intervals as tuples
    pos_intervals, neg_intervals = bedtool_to_intervals(bt, genome)
    seq_lens = seq_lens_from_intervals(pos_intervals + neg_intervals)
    max_seq_len = np.max(seq_lens)
    
    #num_intervals = len(intervals)
    num_pos = len(pos_intervals)
    num_neg = len(neg_intervals)
    
    ## Split intervals into train and test sets
    shuffle(neg_intervals)
    shuffle(pos_intervals)
    
    pos_split_index = int(train_test_split * num_pos)
    neg_split_index = int(train_test_split * num_neg)
    
    pos_train_intervals = pos_intervals[:pos_split_index]
    neg_train_intervals = neg_intervals[:neg_split_index]
    
    pos_test_intervals = pos_intervals[pos_split_index:]
    neg_test_intervals = neg_intervals[neg_split_index:]
    
    
    
    ## Define data generators
    
    train_gen = data_gen_from_intervals(pos_train_intervals,  neg_train_intervals, genome, max_seq_len, 
                                        batch_size = batch_size, 
                                        pad_by = kernel_width, 
                                        aug_by = aug_by)
    
    test_gen = data_gen_from_intervals(pos_test_intervals,  neg_test_intervals, genome, max_seq_len, 
                                        batch_size = batch_size, 
                                        pad_by = kernel_width, 
                                        aug_by = aug_by)
    
    
    ## Define callbacks
    if swa_start is not None:
        swa = SWA(start_epoch=swa_start, 
                  interval=cycle_length)
    else:
        swa = SWA(start_epoch=(num_epochs //2), 
                  interval=cycle_length)
    
    steps_per_epoch = np.ceil((intervals_per_epoch / batch_size))
    validation_steps = int((1 - train_test_split) * steps_per_epoch)
    
    
    schedule = SGDRScheduler(min_lr=1e-5,
                             max_lr=1e-1,
                             steps_per_epoch=steps_per_epoch,
                             lr_decay=1.0,
                             cycle_length=cycle_length,
                             mult_factor=1.0)
    
    
    callbacks_list = [schedule, swa]
    
    ## Construct Model
    model = construct_model(num_kernels=num_kernels, 
                            kernel_width=kernel_width, 
                            seq_len=None,
                            kernel_initializer='glorot_uniform', 
                            optimizer='adam', 
                            activation='linear', 
                            l1_reg=l1_reg, 
                            gaussian_noise = gaussian_noise)
    
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=num_epochs,
                                  validation_data=test_gen,
                                  validation_steps=validation_steps,
                                  callbacks=callbacks_list,
                                  shuffle=True,
                                  use_multiprocessing=False,
                                  workers=1,
                                  max_queue_size=10,
                                  verbose=1)
    
    model_file = output_prefix + ".model.h5"
    model.save(model_file)
    
    bed_file = output_prefix + ".bed"
    hits = scan_intervals_for_kernels(bedgraph, model_file, genome_fasta, output_prefix, scan_pos_only = False)
    with open(bed_file, "w") as f:
        for hit in hits:
            f.write("\t".join([str(x) for x in hit]) + "\n")



    raw_ppms, trimmed_ppms = hits_to_ppms(hits)

    raw_meme_file = output_prefix + ".raw.meme"
    trimmed_meme_file = output_prefix + ".trimmed.meme"

    ppms_to_meme(raw_ppms, raw_meme_file)
    ppms_to_meme(trimmed_ppms, trimmed_meme_file)
    
if __name__ == '__main__':
    main()
