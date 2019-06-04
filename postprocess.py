from pybedtools import BedTool
from pyfaidx import Fasta 
from preprocess import bedtool_to_intervals
from keras.models import load_model
from models import construct_scan_model
from sequence import encode_sequence, decode_sequence
from motif import seq_list_to_ppm, ppms_to_meme, trim_ppm_on_information
import numpy as np

def scan_intervals_for_kernels(bedgraph, model_file, genome_fasta, output_prefix, scan_pos_only = True):

    bt = BedTool(bedgraph)
    model = load_model(model_file)
    genome = Fasta(genome_fasta,
                   as_raw=True,
                   sequence_always_upper=True) 


    conv_weights = model.get_layer('conv1d_1').get_weights()[0]
    kernel_width = conv_weights.shape[0]
    num_kernels = conv_weights.shape[2]
    w2 = kernel_width // 2

    scan_model = construct_scan_model(conv_weights)
    
    ## Convert BedTool object to a list of intervals w/ labels (as tuples)
    pos_intervals, neg_intervals = bedtool_to_intervals(bt, genome)
    
    if scan_pos_only:
        intervals_to_scan = pos_intervals
    else:
        intervals_to_scan = pos_intervals + neg_intervals

    
    num_sites = 0
    hits = []
    for index, interval in enumerate(intervals_to_scan):
        if index % 1000 == 0:
            print("{} sequences processed".format(index))
        chrom, start, stop = interval
        seq = genome[chrom][start:stop]
        padded_seq = kernel_width*'N' + seq + kernel_width*'N'
        encoded_seq = encode_sequence(padded_seq, N = [0.25, 0.25, 0.25, 0.25])
        encoded_seq_rc = encoded_seq[::-1,::-1]
        conv_for = scan_model.predict(np.expand_dims(encoded_seq, axis = 0))[0]
        conv_rc = scan_model.predict(np.expand_dims(encoded_seq[::-1,::-1], axis = 0))[0]
        
        for i in range(num_kernels):
            matches_for = np.argwhere(conv_for[:,i] > 0)
            num_matches_for = matches_for.shape[0]
            for j in range(num_matches_for):
                num_sites += 1
                matched_seq = decode_sequence(encoded_seq[(matches_for[j,0]):(matches_for[j,0]+kernel_width)])
                motif_start = start + matches_for[j,0] - kernel_width
                motif_end = motif_start + kernel_width
                score = conv_for[matches_for[j,0], i]
                hits.append((chrom, motif_start, motif_end, score, "+", matched_seq, output_prefix + "_" + str(i+1)))
                
        
            matches_rc = np.argwhere(conv_rc[:,i] > 0)
            num_matches_rc = matches_rc.shape[0]
            for j in range(num_matches_rc):
                num_sites += 1
                matched_seq = decode_sequence(encoded_seq_rc[(matches_rc[j,0]):(matches_rc[j,0]+kernel_width)])
                motif_end = stop - matches_rc[j,0] + kernel_width
                motif_start = motif_end - kernel_width
                score = conv_rc[matches_rc[j,0], i]
                hits.append((chrom, motif_start, motif_end, score, "-", matched_seq, output_prefix + "_" + str(i+1)))
          
    return hits

def scan_fasta_for_kernels(fasta_file, conv_weights, output_prefix, scan_pos_only = True):
  
    fasta = Fasta(fasta_file,
                   as_raw=True,
                   sequence_always_upper=True)
    
    coords = list(fasta.keys())
    
    #model = load_model(model_file)
    #conv_weights = model.get_layer('conv1d_1').get_weights()[0]
    kernel_width = conv_weights.shape[0]
    num_kernels = conv_weights.shape[2]
    w2 = kernel_width // 2

    scan_model = construct_scan_model(conv_weights)
    num_sites = 0
    hits = []
    for index, seq in enumerate(fasta):
        if index % 1000 == 0:
            print("{} sequences processed".format(index))
        chrom = coords[index].split(":")[0]
        try:
            start = int(coords[index].split(":")[1].split("-")[0])
        except ValueError:
            start = int(float(coords[index].split(":")[1].split("-")[0]))
            print(coords[index])
        try:
            stop = int(coords[index].split(":")[1].split("-")[1])
        except ValueError:
            stop = int(float(coords[index].split(":")[1].split("-")[1]))
        seq = seq[:]
        padded_seq = kernel_width*'N' + seq + kernel_width*'N'
        encoded_seq = encode_sequence(padded_seq, N = [0.25, 0.25, 0.25, 0.25])
        encoded_seq_rc = encoded_seq[::-1,::-1]
        conv_for = scan_model.predict(np.expand_dims(encoded_seq, axis = 0))[0]
        conv_rc = scan_model.predict(np.expand_dims(encoded_seq[::-1,::-1], axis = 0))[0]
        
        for i in range(num_kernels):
            matches_for = np.argwhere(conv_for[:,i] > 0)
            num_matches_for = matches_for.shape[0]
            for j in range(num_matches_for):
                num_sites += 1
                matched_seq = decode_sequence(encoded_seq[(matches_for[j,0]):(matches_for[j,0]+kernel_width)])
                motif_start = start + matches_for[j,0] - kernel_width
                motif_end = motif_start + kernel_width
                score = conv_for[matches_for[j,0], i]
                hits.append((chrom, motif_start, motif_end, score, "+", matched_seq, output_prefix + "_" + str(i+1), coords[index]))
                
        
            matches_rc = np.argwhere(conv_rc[:,i] > 0)
            num_matches_rc = matches_rc.shape[0]
            for j in range(num_matches_rc):
                num_sites += 1
                matched_seq = decode_sequence(encoded_seq_rc[(matches_rc[j,0]):(matches_rc[j,0]+kernel_width)])
                motif_end = stop - matches_rc[j,0] + kernel_width
                motif_start = motif_end - kernel_width
                score = conv_rc[matches_rc[j,0], i]
                hits.append((chrom, motif_start, motif_end, score, "-", matched_seq, output_prefix + "_" + str(i+1), coords[index]))
          
    return hits



def hits_to_ppms(hits):
    motif_ids = sorted(set([hit[6] for hit in hits]))
    raw_ppms = []
    trimmed_ppms = []
    
    for motif_id in motif_ids:
        seq_list = [hit[5] for hit in hits if hit[6] == motif_id]
        num_seqs = len(seq_list)
        
        print(motif_id)
        print(seq_list[0])
        print(num_seqs)
        raw_ppm = seq_list_to_ppm(seq_list)
        trimmed_ppm = trim_ppm_on_information(raw_ppm, min_info = 0.25)
        raw_ppms.append((motif_id, num_seqs, raw_ppm))
        trimmed_ppms.append((motif_id, num_seqs, trimmed_ppm))
        
    return raw_ppms, trimmed_ppms


# fasta_file = "371.pos.fasta"
# model_file = "371.model.h5"
# output_prefix = "371"
# hits = scan_fasta_for_kernels(fasta_file, model_file, output_prefix, scan_pos_only = True)
#bedgraph = "371.bg" 
#model_file = "371.model.h5"
#genome_fasta = "/project/umw_zhiping_weng/andrewsg/genome/hg38/hg38.fa"
#output_prefix = "371"
# bed_file = output_prefix + ".bed"
#hits = scan_intervals_for_kernels(bedgraph, model_file, genome_fasta, output_prefix, scan_pos_only = False)      
# with open(bed_file, "w") as f:
#        for hit in hits:
#            f.write("\t".join([str(x) for x in hit]) + "\n")

##
#
# raw_ppms, trimmed_ppms = hits_to_ppms(hits)

# raw_meme_file = output_prefix + ".raw.meme"
# trimmed_meme_file = output_prefix + ".trimmed.meme"

# ppms_to_meme(raw_ppms, raw_meme_file)
# ppms_to_meme(trimmed_ppms, trimmed_meme_file)
                        
