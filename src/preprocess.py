import numpy as np
from pyfaidx import Fasta
from src.sequence import encode_sequence
from src.dinuclShuffle import dinuclShuffle

def get_format(bed_file):
    with open(bed_file, "r") as f:
        first_line = f.readline()
        split = first_line.strip().split()
        num_entries = len(split)
        if num_entries == 3:
            print("Bed File provided")
            return "bed"
        elif num_entries == 4:
            print("Bedgraph File provdided")
            return "bg"
        elif num_entries == 10:
            print("narrowPeak file provided")
            return "peak"
        else:
            print("Unrecognized file format provided")
            return None
            
def bed_to_seqs(bed_file, genome_fasta, seq_len = None, store_encoded=False, negs_from="shuffle", weight_samples=True):
    fmt = get_format(bed_file)
    genome = Fasta(genome_fasta, as_raw=True, sequence_always_upper=True)
    seqs = []
    
    with open(bed_file, "r") as f:
        for line in f: 
            if fmt == "bed":
                chrom, start, stop = line.strip().split()[:3]
            elif fmt == "peak":
                chrom, start, stop, _, _, _, signal, _, _, offset = line.strip().split()
            else:
                chrom, start, stop, signal = line.strip().split()
            
            start = int(start)
            stop = int(stop)
            
            if fmt == "bed":
                offset = 0
                signal = 1
            elif fmt == "peak":
                signal = float(signal)
                offset = int(offset)
            else:
                signal = float(signal)
                offset = 0
            
            if fmt in ["bed", "bg"]:
                if seq_len is not None:
                    center = (stop - start) // 2 + start
                    w_2 = seq_len // 2
                    seq_start = center - w_2
                    seq_stop = center + w_2
                else:
                    seq_start = start
                    seq_stop = stop
            else:
                if seq_len is not None:
                    summit = start + offset
                    w_2 = seq_len // 2
                    seq_start = summit - w_2
                    seq_stop = summit + w_2
                else:
                    seq_start = start
                    seq_stop = stop
                    
            coord = chrom + ":" + str(seq_start) + "-" + str(seq_stop)
            try:
                pos_seq = genome[chrom][seq_start:seq_stop]
                if negs_from == "shuffle":
                    neg_seq = dinuclShuffle(pos_seq)
                    if store_encoded:
                        seqs.append([encode_sequence(pos_seq, useN = "uniform"), 1, 1, signal, coord])
                        seqs.append([encode_sequence(neg_seq, useN = "uniform"), 0, 1, signal, coord])
                    else:
                        seqs.append([pos_seq, 1, 1, signal, coord])
                        seqs.append([neg_seq, 0, 1, signal, coord])
                else:
                    if seq_len is not None:
                        left_seq = genome[chrom][seq_start-seq_len:seq_start]
                        right_seq = genome[chrom][seq_stop:seq_stop+seq_len]
                    else:
                        left_seq = genome[chrom][seq_start-(seq_stop-seq_start):seq_start]
                        right_seq = genome[chrom][seq_stop:seq_stop+(seq_stop-seq_start)]
                    
                    if store_encoded:
                        seqs.append([encode_sequence(pos_seq, useN = "uniform"), 1, 1, signal, coord])
                        seqs.append([encode_sequence(left_seq, useN = "uniform"), 0, 1, signal, coord])
                        seqs.append([encode_sequence(right_seq, useN = "uniform"), 0, 1, signal, coord])
                    else:
                        seqs.append([pos_seq, 1, 1, signal, coord])
                        seqs.append([left_seq, 0, 1, signal, coord])
                        seqs.append([right_seq, 0, 1, signal, coord])
                        
                        
            except KeyError:
                print("{} not in genome fasta".format(coord))
            
                
    for seq in seqs:
        if store_encoded:
            if seq[0].shape[0] == 0:
                seqs.remove(seq)
        else:
            if len(seq[0]) == 0:
                seqs.remove(seq)
    
    if fmt in ["bg", "peak"] and weight_samples == True:
        print("Weighting samples by rank")
        seqs.sort(key= lambda seq: seq[3], reverse=True)
        num_seqs = len(seqs)
        max_signal = seqs[0][3]
        for i, seq in enumerate(seqs):
            seqs[i][2] = seqs[i][3] / max_signal
    return seqs
 
def fasta_to_seqs(pos_fasta, seq_len = None, neg_fasta = None, store_encoded=True):
    seqs = []
    with open(pos_fasta) as f:
        for i, line in enumerate(f):
            if i % 2 == 0:
                coord = line.strip().split()[0][1:]
                
            else:
                pos_seq = line.strip().split()[0].upper()
                if store_encoded:
                    seqs.append([encode_sequence(pos_seq, useN = "uniform"), 1, 1, 1, coord])
                else:
                    seqs.append([pos_seq, 1, 1, 1, coord])
                    
                if neg_fasta is None:
                    neg_seq = dinuclShuffle(pos_seq)
                    if store_encoded:
                        seqs.append([encode_sequence(neg_seq, useN = "uniform"), 0, 1, 1, coord])
                    else:
                        seqs.append([neg_seq, 0, 1, 1, coord])
    
    if neg_fasta is not None:
        with open(neg_fasta) as f:
            for i, line in enumerate(f):
                if i % 2 == 0:
                    coord = line.strip().split()[0][1:]

                else:
                    neg_seq = line.strip().split()[0].upper()
                    if store_encoded:
                        seqs.append([encode_sequence(neg_seq, useN = "uniform"), 0, 1, 1, coord])
                    else:
                        seqs.append([neg_seq, 0, 1, 1, coord])
    return(seqs)
                
                
        
        
