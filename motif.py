from sequence import encode_sequence
import numpy as np

def seq_list_to_ppm(seq_list):
    encoded_seq_list = [encode_sequence(seq, N = [0, 0, 0, 0]) for seq in seq_list]
    encoded_seq_array = np.array(encoded_seq_list)
    pfm = np.sum(encoded_seq_array, axis=0)
    if np.min(np.sum(pfm, axis=1)) > 0:
        ppm = pfm / np.sum(pfm, axis=1).reshape((pfm.shape[0],1))
    else:
        pfm += 0.25
        ppm = pfm / np.sum(pfm, axis=1).reshape((pfm.shape[0],1))
    return ppm

def pfm_to_ppm(pfm):
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
    return ppm[start_idx:end_idx,:]

def ppms_to_meme(ppms, output_file):
    with open(output_file, "w") as f:
        f.write("MEME version 4 \n\n")
        f.write("ALPHABET= ACGT \n\n")
        f.write("strands: + - \n\n")
        f.write("Background letter frequencies \n")
        f.write("A 0.25 C 0.25 G 0.25 T 0.25 \n\n")
        for ppm_tup in ppms:
            motif_id, n_sites, ppm = ppm_tup
            f.write("MOTIF " + motif_id + "\n")
            f.write("letter-probability matrix: alength= 4 w= {} nsites= {} \n".format(str(ppm.shape[0]), n_sites))
            for j in range(ppm.shape[0]):
                tempPpm = ppm[j,:]
                f.write("\t".join([str(x) for x in tempPpm]))
                f.write("\n")
            f.write("\n")