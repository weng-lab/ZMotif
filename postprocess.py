from pybedtools import BedTool
from pyfaidx import Fasta 
from models import construct_scan_model
from sequence import encode_sequence, decode_sequence
from motif import seq_list_to_ppm, ppms_to_meme, trim_ppm_on_information
import numpy as np
from utils import progress
def scan_fasta_for_kernels(fasta_file, conv_weights, output_prefix, mode = 'anr', scan_pos_only = True):
  
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
            # update progress bar
            progress(index, len(coords), "scanning sequences")
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
            if mode == 'anr':
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
            else:
                if np.max(conv_for[:,i]) > 0:
                    match_for = np.argmax(conv_for[:,i] > 0)
                    num_sites += 1
                    matched_seq = decode_sequence(encoded_seq[match_for-2:(match_for+kernel_width+2)])
                    motif_start = start + match_for - kernel_width
                    motif_end = motif_start + kernel_width
                    score = conv_for[match_for, i]
                    hits.append((chrom, motif_start, motif_end, score, "+", matched_seq, output_prefix + "_" + str(i+1), coords[index]))
                        
                    
                if np.max(conv_rc[:,i]) > 0:
                    match_rc = np.argmax(conv_rc[:,i] > 0)
                    num_sites += 1
                    matched_seq = decode_sequence(encoded_seq_rc[match_rc-2:(match_rc+kernel_width+2)])
                    motif_end = stop - match_rc + kernel_width
                    motif_start = motif_end - kernel_width
                    score = conv_rc[match_rc, i]
                    hits.append((chrom, motif_start, motif_end, score, "-", matched_seq, output_prefix + "_" + str(i+1), coords[index]))
                    
                
          
    return hits


# def vectorized_app(arr_1,arr_2):
#     W = arr_1.shape[0] # Window size
#     idx = np.arange(arr_2.shape[0]-W+1)[:,None] + np.arange(W)
#     return arr_1*arr_2[idx]


# def scan_fasta_for_kernels_np(fasta_file, conv_weights, output_prefix, mode = 'anr', scan_pos_only = True):
  
#     fasta = Fasta(fasta_file,
#                    as_raw=True,
#                    sequence_always_upper=True)
    
#     coords = list(fasta.keys())
    
#     kernel_width = conv_weights.shape[0]
#     num_kernels = conv_weights.shape[2]
#     w2 = kernel_width // 2
    
#     kernel = conv_weights[:,:,24]
#     hits = []
#     num_hits = 0
#     for index, seq in enumerate(fasta):
#         if index % 1000 == 0:
#             print("{} sequences processed".format(index))
#         chrom = coords[index].split(":")[0]
#         try:
#             start = int(coords[index].split(":")[1].split("-")[0])
#         except ValueError:
#             start = int(float(coords[index].split(":")[1].split("-")[0]))
#             print(coords[index])
#         try:
#             stop = int(coords[index].split(":")[1].split("-")[1])
#         except ValueError:
#             stop = int(float(coords[index].split(":")[1].split("-")[1]))
#         seq = seq[:]
#         padded_seq = kernel_width*'N' + seq + kernel_width*'N'
#         encoded_seq = encode_sequence(padded_seq, N = [0.25, 0.25, 0.25, 0.25])
#         encoded_seq_rc = encoded_seq[::-1,::-1]
#         if np.max(convolve(encoded_seq, kernel, mode="constant", cval=0.25)) > 0:
#             num_hits += 1
#         if np.max(convolve(encoded_seq_rc, kernel, mode="constant", cval=0.25)) > 0:
#             num_hits += 1
#     print(num_hits)

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


# def read_hdf5(path):

#     weights = {}

#     keys = []
#     with h5py.File(path, 'r') as f: # open file
#         f.visit(keys.append) # append all keys to list
#         for key in keys:
#             if ':' in key: # contains data if ':' in key
#                 print(f[key].name)
#                 weights[f[key].name] = f[key].value
#     return weights


# fasta_file = "/project/umw_zhiping_weng/andrewsg/motif_finder/cistrome/results/371/371.pos.fasta"
# model_weights = read_hdf5("/project/umw_zhiping_weng/andrewsg/motif_finder/cistrome/results/371/371.weights.h5")
# conv_weights = model_weights['/conv1d_1/conv1d_1/kernel:0']
# output_prefix = "test"
# hits = scan_fasta_for_kernels_np(fasta_file, conv_weights, output_prefix, scan_pos_only = True)
# raw_ppms, trimmed_ppms = hits_to_ppms(hits)
# raw_meme_file = output_prefix + ".raw.meme"
# ppms_to_meme(raw_ppms, raw_meme_file)
