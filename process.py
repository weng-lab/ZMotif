from utils import fasta_to_array, fimo, get_motif_matches, encode_sequence, ppms_to_meme
from keras.models import load_model
import numpy as np
from pyfaidx import Fasta
import json
## load model

def vectorized_app(arr_1,arr_2):
    W = arr_1.shape[0] 
    idx = np.arange(arr_2.shape[0]-W+1)[:,None] + np.arange(W)
    return arr_1*arr_2[idx]

def get_decoded_seq(encoded_seq):
    seq_list = encoded_seq.astype('int').tolist()
    decoded_seq_list = []
    for nuc in seq_list:
        if nuc == [1, 0, 0, 0]:
            decoded_seq_list.append('A')
        elif nuc == [0, 1, 0, 0]:
            decoded_seq_list.append('C')
        elif nuc == [0, 0, 1, 0]:
            decoded_seq_list.append('G')
        elif nuc == [0, 0, 0, 1]:
            decoded_seq_list.append('T')
        else:
            decoded_seq_list.append('N')

    return "".join(decoded_seq_list)

def seq_list_to_ppm(seq_list):
    encoded_seq_list = [encode_sequence(seq, N = [0, 0, 0, 0]) for seq in seq_list]
    encoded_seq_array = np.array(encoded_seq_list)
    pfm = np.sum(encoded_seq_array, axis=0)
    ppm = pfm / np.sum(pfm, axis=1).reshape((pfm.shape[0],1))
    return ppm

def ppm_to_pwm(ppm, background = [0.25, 0.25, 0.25, 0.25]):
    pwm = np.zeros(ppm.shape)
    w = ppm.shape[0]
    for i in range(w):
        for j in range(4):
            pwm[i,j] = np.log2((ppm[i,j] + .001) / 0.25)
    return pwm

def calculate_information_content(ppm):
    w = ppm.shape[0]
    info = np.zeros(w)
    for i in range(w):
        for j in range(4):
            info[i] += ppm[i,j] * np.log2((ppm[i,j] + .001) / 0.25)
    return info

def trim_ppm_on_information(ppm, min_info = 0.1):
    w = ppm.shape[0]
    info = calculate_information_content(ppm)
    print(info)
    start_idx = 0
    end_idx = w
    for i in range(w):
        if info[i] < min_info:
            start_idx += 1
        else:
            break
    
    for i in range(w):
        if info[-i-1] < min_info:
            end_idx -= 1
        else:
            break
    print(start_idx)
    print(end_idx)
    return ppm[start_idx:end_idx,:]

def motif_scan(model_file, fasta_files, output_prefix, min_num_seqs = 100, output_bed = True):
    ## get number of sequences to scan
    num_seqs = 0
    num_fasta_files = len(fasta_files)
    for fasta_file in fasta_files:
        fasta = Fasta(fasta_file, as_raw=True, sequence_always_upper=True)
        for seq in fasta:
            num_seqs += 1
    print("Scanning {0} sequences from {1} fasta files".format(num_seqs, num_fasta_files))
    ## score sequences
    model = load_model(model_file)
    conv_weights = model.get_layer('conv1d_1').get_weights()[0]
    num_kernels = conv_weights.shape[2]
    seqs_per_kernel = np.zeros(num_kernels).astype('int')
    for fasta_file in fasta_files:
        temp_seq_array = fasta_to_array(fasta_file, pad_by = 0, zero_background = False)
        scores = fimo(temp_seq_array, conv_weights)
        scores[scores > 0] = 1
        scores[scores <= 0] = 0
        seqs_per_kernel += np.sum(scores, axis = 0).astype('int')
    print(seqs_per_kernel)
    
    kernels_to_scan = np.ravel(np.argwhere(seqs_per_kernel >= min_num_seqs))
    num_kernels_to_scan = kernels_to_scan.shape[0]
    filtered_conv_weights = conv_weights[:,:,kernels_to_scan]
    np.save(output_prefix + ".filtered_conv_weights.npy", filtered_conv_weights)
 
    print(filtered_conv_weights.shape)
    print("Scanning sequences for {} motifs".format(num_kernels_to_scan))
    ppms = []
    for i in range(num_kernels_to_scan):
        num_matches = 0
        matches = []
        coords = []
        scores = []
        kernel = filtered_conv_weights[:,:,i]
        kernel_width = kernel.shape[0]
        for fasta_file in fasta_files:
            fasta = Fasta(fasta_file, as_raw=True, sequence_always_upper=True)
            for seq in fasta:
                
                name = seq.name
                chrom = name.split(":")[0]
                seq_start = int(name.split(":")[1].split("-")[0])
                seq_end = int(name.split(":")[1].split("-")[1])
                padded_seq = kernel_width * 'N' + seq[:] + kernel_width * 'N'
    
                encoded_seq = encode_sequence(padded_seq, N = [0.25, 0.25, 0.25, 0.25])
                encoded_seq_rc = encode_sequence(padded_seq, N = [0.25, 0.25, 0.25, 0.25])[::-1,::-1]
    
                conv_seq = np.sum(vectorized_app(kernel, encoded_seq), axis=(1,2))
                conv_seq_rc = np.sum(vectorized_app(kernel, encoded_seq_rc), axis=(1,2))

                matches_for = np.argwhere(conv_seq > 0)
                num_matches_for = matches_for.shape[0]
                for j in range(num_matches_for):
                    matched_seq = get_decoded_seq(encoded_seq[matches_for[j,0]:matches_for[j,0]+kernel_width])
                    num_matches += 1
                    motif_start = seq_start + matches_for[j,0] - kernel_width
                    motif_end = motif_start + kernel_width
                    matches.append(matched_seq)
                    coords.append((chrom, motif_start, motif_end, "+"))
                    score = conv_seq[matches_for[j,0]]
                    scores.append(score)
                    #print(motif_start, motif_end, matched_seq)
                    #quit()
                matches_rc = np.argwhere(conv_seq_rc > 0)
                num_matches_rc = matches_rc.shape[0]
                for j in range(num_matches_rc):
                    matched_seq = get_decoded_seq(encoded_seq_rc[matches_rc[j,0]:matches_rc[j,0]+kernel_width])
                    matches.append(matched_seq)
                    motif_start = seq_end - matches_rc[j,0]
                    motif_end = seq_end - matches_rc[j,0] + kernel_width
                    num_matches += 1
                    coords.append((chrom, motif_start, motif_end, "-"))
                    score = conv_seq_rc[matches_rc[j,0]]
                    scores.append(score)
                    #print(motif_start, motif_end, matched_seq)
        if output_bed:
            with open(output_prefix + ".bed", "a") as f:    
                for coord, score, match in zip(coords, scores, matches):
                    f.write("\t".join([coord[0], str(coord[1]), str(coord[2]), str(score), coord[3], match, output_prefix + "_" + str(i+1)]) + "\n")
        
        ppm = seq_list_to_ppm(matches)
        ppms.append((ppm, num_matches))
    ppms_to_meme(ppms, output_prefix + '.meme', output_prefix)
    trimmed_ppms = []
    for ppm_tup in ppms:
        ppm = ppm_tup[0]
        nsites = ppm_tup[1]
        trimmed_ppms.append((trim_ppm_on_information(ppm, min_info = 0.1), nsites))
    ppms_to_meme(trimmed_ppms, output_prefix + '.trimmed.meme', output_prefix)
    np.save(output_prefix + ".ppms.npy", np.array([ppm[0] for ppm in ppms]))
    with open(output_prefix + ".ppms.json", "w") as f:
        json.dump(np.array([ppm[0].tolist() for ppm in ppms]).tolist(), f)

motif_scan("371.model.h5", ["371.pos.fasta"], '371', min_num_seqs = 100)
#ppms = np.load("HNF4A_HepG2.ppms.npy")
#print(ppms.shape)
#test_ppm = ppms[4,:,:]
#trimmed_ppm = trim_ppm_on_information(test_ppm)
#pwm = ppm_to_pwm(trimmed_ppm)
#print(pwm)
#with open("HNF4A_HepG2" + ".ppms.json", "w") as f:
#    json.dump(ppms.tolist(), f) 
