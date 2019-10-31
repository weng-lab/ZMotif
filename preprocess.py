import numpy as np
from pyfaidx import Fasta
from pybedtools import BedTool
from sequence import encode_sequence
from dinuclShuffle import dinuclShuffle

def bg_to_seqs(bg, fasta, max_seq_len, store_encoded=False):
           
    genome = Fasta(fasta, as_raw=True, sequence_always_upper=True)
    seqs = []
    with open(bg, "r") as f:
        for line in f:
            chrom, start, stop, score = line.strip().split()
            start = int(start)
            stop = int(stop)
            score = float(score)
            input_seq_len = stop - start
            if input_seq_len > max_seq_len:
                # Get center of positive sequence
                new_start = start + input_seq_len//2 - (max_seq_len // 2)
                new_stop = new_start + max_seq_len
                coord = chrom + ":" + str(new_start) + "-" + str(new_stop)
                try:
                    pos_seq = genome[chrom][new_start:new_stop]
                    left_seq = genome[chrom][start-max_seq_len:start]
                    right_seq = genome[chrom][stop:stop+max_seq_len]
                    if store_encoded:
                        seqs.append([encode_sequence(pos_seq), 1, 1, score, coord])
                        seqs.append([encode_sequence(left_seq), 0, 1, score, coord])
                        seqs.append([encode_sequence(right_seq), 0, 1, score, coord])
                    else:
                        seqs.append([pos_seq, 1, 1, score, coord])
                        seqs.append([left_seq, 0, 1, score, coord])
                        seqs.append([right_seq, 0, 1, score, coord])
                except KeyError:
                    pass
            else:
                try:
                    coord = chrom + ":" + str(start) + "-" + str(stop)
                    pos_seq = genome[chrom][start:stop]
                    left_seq = genome[chrom][start-input_seq_len:start]
                    right_seq = genome[chrom][stop:stop+input_seq_len]
                    if store_encoded:
                        seqs.append([encode_sequence(pos_seq), 1, 1, score, coord])
                        seqs.append([encode_sequence(left_seq), 0, 1, score, coord])
                        seqs.append([encode_sequence(right_seq), 0, 1, score, coord])
                    else:
                        seqs.append([genome[chrom][start:stop], 1, 1, score, coord])
                        seqs.append([genome[chrom][start-input_seq_len:start], 0, 1, score, coord])
                        seqs.append([genome[chrom][stop:stop+input_seq_len], 0, 1, score, coord])
                except KeyError:
                    pass
                
    for seq in seqs:
        if store_encoded:
            if seq[0].shape[0] == 0:
                seqs.remove(seq)
        else:
            if len(seq[0]) == 0:
                seqs.remove(seq)

    return seqs

def narrowpeak_to_seqs(peak_file, genome_fasta, seq_len, store_encoded=False, negs_from="shuffle"):
    genome = Fasta(genome_fasta, as_raw=True, sequence_always_upper=True)
    seqs = []
    with open(peak_file, "r") as f:
        for line in f:
            chrom, start, stop, _, _, _, signal, _, _, offset = line.strip().split()
            start = int(start)
            stop = int(stop)
            signal = float(signal)
            offset = int(offset)
            center = (stop - start) // 2 + start
            summit = start + offset
            w_2 = seq_len // 2
            seq_start = summit - w_2
            seq_stop = summit + w_2
            coord = chrom + ":" + str(seq_start) + "-" + str(seq_stop)
            try:
                pos_seq = genome[chrom][seq_start:seq_stop]
                neg_seq = dinuclShuffle(pos_seq)
                if store_encoded:
                    seqs.append([encode_sequence(pos_seq), 1, 1, signal, coord])
                    seqs.append([encode_sequence(neg_seq), 0, 1, signal, coord])
                else:
                    seqs.append([pos_seq, 1, 1, signal, coord])
                    seqs.append([neg_seq, 0, 1, signal, coord])
            except KeyError:
                print("{} not in genome fasta".format(coord))
            
                
    for seq in seqs:
        if store_encoded:
            if seq[0].shape[0] == 0:
                seqs.remove(seq)
        else:
            if len(seq[0]) == 0:
                seqs.remove(seq)

    return seqs