from pyfaidx import Fasta 
from src.models import construct_scan_model
from src.sequence import encode_sequence, decode_sequence
from src.motif import seq_list_to_ppm, ppms_to_meme, trim_ppm_on_information
import numpy as np
from src.utils import progress
import sys

def scan_seqs_for_kernels(seqs, conv_weights, output_prefix, mode = 'anr', scan_pos_only = True, store_encoded=False, expand_by=2):
    
    if scan_pos_only:
        seqs_to_scan = [seq for seq in seqs if seq[1] == 1]
    else:
        seqs_to_scan = seqs
        
    kernel_width = conv_weights.shape[0]
    num_kernels = conv_weights.shape[2]
    w2 = kernel_width // 2
    
    scan_model = construct_scan_model(conv_weights)
    num_sites = 0
    hits = []
    for index, seq in enumerate(seqs_to_scan):
        if index % 1000 == 0:
            # update progress bar
            progress(index, len(seqs_to_scan), "Scanning sequences")

        
        if store_encoded:
            encoded_seq = np.vstack((0.25*np.ones((kernel_width,4)), seq[0], 0.25*np.ones((kernel_width,4))))
        else:
            encoded_seq = np.vstack((0.25*np.ones((kernel_width,4)), encode_sequence(seq[0]), 0.25*np.ones((kernel_width,4))))
#         encoded_seq = np.vstack((np.zeros((kernel_width,4)), seq[0], np.zeros((kernel_width,4))))

        encoded_seq_rc = encoded_seq[::-1,::-1]
        coord = seq[4]
        start = int(float(coord.split(":")[1].split("-")[0]))
        stop = int(float(coord.split(":")[1].split("-")[1]))
        chrom = coord.split(":")[0]
        conv_for = scan_model.predict(np.expand_dims(encoded_seq, axis = 0))[0]
        conv_rc = scan_model.predict(np.expand_dims(encoded_seq_rc, axis = 0))[0]
        
        for i in range(num_kernels):
            if np.max(conv_for[:,i]) > 0 or np.max(conv_rc[:,i]) > 0:
        
#         for i in range(num_kernels):
                if np.max(conv_for[:,i]) > np.max(conv_rc[:,i]):
                    match_for = np.argmax(conv_for[:,i] > 0)
                    num_sites += 1
                    matched_seq = decode_sequence(encoded_seq[match_for-expand_by:(match_for+kernel_width+expand_by)])
                    motif_start = start + match_for - kernel_width
                    motif_end = motif_start + kernel_width
                    score = conv_for[match_for, i]
                    hits.append((chrom, motif_start, motif_end, score, "+", matched_seq, output_prefix + "_" + str(i+1), coord))
                else:
                    match_rc = np.argmax(conv_rc[:,i] > 0)
                    num_sites += 1
                    matched_seq = decode_sequence(encoded_seq_rc[match_rc-expand_by:(match_rc+kernel_width+expand_by)])
                    motif_end = stop - match_rc + kernel_width
                    motif_start = motif_end - kernel_width
                    score = conv_rc[match_rc, i]
                    hits.append((chrom, motif_start, motif_end, score, "-", matched_seq, output_prefix + "_" + str(i+1), coord))
                    
    progress(len(seqs_to_scan), len(seqs_to_scan), "Scanning sequences")
    sys.stdout.write("\n")
    return hits


def hits_to_ppms(hits):
    motif_ids = sorted(set([hit[6] for hit in hits]))
    raw_ppms = []
    trimmed_ppms = []
    
    for motif_id in motif_ids:
        seq_list = [hit[5] for hit in hits if hit[6] == motif_id]
        num_seqs = len(seq_list)
        
        raw_ppm = seq_list_to_ppm(seq_list)
        trimmed_ppm = trim_ppm_on_information(raw_ppm, min_info = 0.25)
        raw_ppms.append((motif_id, num_seqs, raw_ppm))
        trimmed_ppms.append((motif_id, num_seqs, trimmed_ppm))
        
    return raw_ppms, trimmed_ppms

import h5py
def read_hdf5(path):

    weights = {}

    keys = []
    with h5py.File(path, 'r') as f: # open file
        f.visit(keys.append) # append all keys to list
        for key in keys:
            if ':' in key: # contains data if ':' in key
                print(f[key].name)
                weights[f[key].name] = f[key].value
    return weights


# from preprocess import bg_to_seqs, bg_to_fasta
# import h5py

# seqs = bg_to_seqs("/data/zusers/andrewsg/motif_discovery/371.bg", "/home/andrewsg/genome/hg38/hg38.fa", 500, store_encoded=True)
# bg_to_fasta("/data/zusers/andrewsg/motif_discovery/371.bg", "/home/andrewsg/genome/hg38/hg38.fa", "test.fasta")
# model_weights = read_hdf5("/data/zusers/andrewsg/motif_discovery/results/1448/1448.weights.h5")
# print(model_weights)
# conv_weights = model_weights['/conv1d_1/conv1d_1/kernel:0']
# output_prefix = "test"
# hits = scan_seqs_for_kernels(seqs, conv_weights, output_prefix, scan_pos_only = True)
# hits = scan_fasta_for_kernels("test.fasta", conv_weights, output_prefix, scan_pos_only = True)
# for hit in hits:
#     print(hit[5])
# raw_ppms, trimmed_ppms = hits_to_ppms(hits)
# raw_meme_file = output_prefix + ".raw.meme"
# ppms_to_meme(raw_ppms, raw_meme_file)
