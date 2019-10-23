from inputs import get_args 
from models import construct_model
from preprocess import get_sequences_from_bedgraph
from data_generators import DataGenerator
from AdamW import AdamW
from custom_callbacks import SGDRScheduler, SWA
import numpy as np
from pybedtools import BedTool
from postprocess import *
import random

def main():
    args = get_args()
    bedgraph = args.bedgraph
    genome = args.genome
    
    seed = 12
    epochs = 100
    max_seq_len = 500
    batch_size = 32
    train_test_split = 0.9
    seqs_per_epoch = 5000
    
    
    seq_list = get_sequences_from_bedgraph(bedgraph, genome, max_seq_len=500)
    random.seed(12)
    random.shuffle(seq_list)
    
    train_seqs = seq_list[:int(train_test_split * len(seq_list))]
    test_seqs = seq_list[int(train_test_split * len(seq_list)):]
    
    train_seqs.sort(reverse=True, key = lambda x: x.score)
    
    if seqs_per_epoch > 0:
        steps_per_epoch = int(np.ceil(seqs_per_epoch / batch_size))
    else:
        steps_per_epoch = int(np.ceil(len(train_seqs) / batch_size))
    
    train_gen = DataGenerator(train_seqs, max_seq_len, batch_size=batch_size, pad_by=32, seqs_per_epoch=seqs_per_epoch)
    test_gen = DataGenerator(test_seqs, max_seq_len, batch_size=batch_size, pad_by=32)
    
    adamw = AdamW(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0., weight_decay=0.025, batch_size=batch_size, samples_per_epoch=(batch_size * steps_per_epoch), epochs=epochs)
    
    model = construct_model(num_kernels=16, optimizer=adamw)
    
    schedule = SGDRScheduler(min_lr=.01,
                             max_lr=.1,
                             steps_per_epoch=steps_per_epoch,
                             lr_decay=1.0,
                             cycle_length=1,
                             mult_factor=1.0)
    
    swa = SWA(epochs)
    model.fit_generator(train_gen,
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        callbacks=[schedule],
                        class_weight={0 : 1, 1 : 2})
                        
                                                
    
    conv_weights = model.get_layer('conv1d_1').get_weights()[0]
    output_prefix = "test"
    
    
    pos_fasta = output_prefix + ".pos.fasta"
    bt = BedTool(bedgraph)
    bt.sequence(fi=genome)
    with open(pos_fasta, "w") as fasta:
        print(open(bt.seqfn).read(), file=fasta)
        
    bed_file = output_prefix + ".bed"
    pos_hits = scan_fasta_for_kernels(pos_fasta, conv_weights, output_prefix, mode = "anr", scan_pos_only = True)
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