import sys
import numpy as np
from sequence import encode_sequence
from motif import pfm_to_ppm, calculate_information_content

bed_file = sys.argv[1]
min_info = float(sys.argv[2])
trimmed_bed = sys.argv[3]

num_lines = 0
with open(bed_file) as f:
    for line in f:
        if num_lines == 0:
            split = line.strip().split()
            w = len(split[5])
        num_lines += 1
        
motif_ids = ["" for i in range(num_lines)]

with open(bed_file) as f:
    for i, line in enumerate(f):
        motif_ids[i] = line.strip().split()[6]
        
motif_ids = list(set(motif_ids))
print(motif_ids)
for motif_id in motif_ids:
    pfm = np.zeros((w,4))
    with open(bed_file) as f:
        for line in f:
            if line.strip().split()[6] == motif_id:
            #print(encode_sequence(line.strip().split()[5]).shape)
                pfm += encode_sequence(line.strip().split()[5])
        
    ppm = pfm_to_ppm(pfm)
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
    #print(motif_id, start_idx, end_idx)
    left_offset = start_idx
    right_offset = w - end_idx
    #print(left_offset, right_offset)
    
    with open(bed_file, "r") as f, open(trimmed_bed, "a+") as output:
        for line in f:
            chrom, start, end, score, strand, seq, kernel_id, coord = line.strip().split()
            if kernel_id == motif_id:
                if strand == "+":
                    new_start = int(start) + left_offset
                    new_end = int(end) - right_offset
                    #print(new_end - new_start)
                else:
                    new_start = int(start) + right_offset
                    new_end = int(end) - left_offset
                    #print(new_end - new_start)
                
                
                new_seq = seq[start_idx:end_idx]
                
                output.write("\t".join([chrom, str(new_start), str(new_end), score, strand, new_seq, kernel_id, coord]) + "\n")
